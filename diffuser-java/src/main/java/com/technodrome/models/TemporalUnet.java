package com.technodrome.models;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv1d;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import com.technodrome.models.helpers.Conv1dBlock;
import com.technodrome.models.helpers.Downsample1d;
import com.technodrome.models.helpers.Mish;
import com.technodrome.models.helpers.SinusoidalPosEmb;
import com.technodrome.models.helpers.Upsample1d;

import java.util.ArrayList;
import java.util.List;

/**
 * Temporal U-Net architecture for denoising trajectories.
 * Uses 1D convolutions for temporal processing with skip connections.
 */
public class TemporalUnet extends AbstractBlock {

    private static final byte VERSION = 1;

    private final int transitionDim;
    private final int dim;
    private final int[] dimMults;

    private final SequentialBlock timeMlp;
    private final List<Block[]> downs;
    private final ResidualTemporalBlock midBlock1;
    private final ResidualTemporalBlock midBlock2;
    private final List<Block[]> ups;
    private final SequentialBlock finalConv;

    public TemporalUnet(int horizon, int transitionDim, int condDim) {
        this(horizon, transitionDim, condDim, 32, new int[]{1, 2, 4, 8});
    }

    public TemporalUnet(int horizon, int transitionDim, int condDim, int dim, int[] dimMults) {
        super(VERSION);

        this.transitionDim = transitionDim;
        this.dim = dim;
        this.dimMults = dimMults;

        // Calculate channel dimensions
        int[] dims = new int[dimMults.length + 1];
        dims[0] = transitionDim;
        for (int i = 0; i < dimMults.length; i++) {
            dims[i + 1] = dim * dimMults[i];
        }

        // Create in/out pairs
        int[][] inOut = new int[dimMults.length][2];
        for (int i = 0; i < dimMults.length; i++) {
            inOut[i][0] = dims[i];
            inOut[i][1] = dims[i + 1];
        }

        System.out.println("[ models/temporal ] Channel dimensions:");
        for (int[] pair : inOut) {
            System.out.println("  " + pair[0] + " -> " + pair[1]);
        }

        int timeDim = dim;

        // Time MLP: SinusoidalPosEmb -> Linear -> Mish -> Linear
        this.timeMlp = new SequentialBlock();
        timeMlp.add(new SinusoidalPosEmb(dim));
        timeMlp.add(Linear.builder().setUnits(dim * 4).build());
        timeMlp.add(new Mish());
        timeMlp.add(Linear.builder().setUnits(dim).build());
        addChildBlock("timeMlp", timeMlp);

        // Downsampling path
        this.downs = new ArrayList<>();
        int currentHorizon = horizon;
        int numResolutions = inOut.length;

        for (int ind = 0; ind < numResolutions; ind++) {
            int dimIn = inOut[ind][0];
            int dimOut = inOut[ind][1];
            boolean isLast = ind >= numResolutions - 1;

            Block[] downBlock = new Block[3];
            downBlock[0] = new ResidualTemporalBlock(dimIn, dimOut, timeDim, currentHorizon);
            downBlock[1] = new ResidualTemporalBlock(dimOut, dimOut, timeDim, currentHorizon);
            downBlock[2] = isLast ? Blocks.identityBlock() : new Downsample1d(dimOut);

            addChildBlock("down_" + ind + "_resnet1", (AbstractBlock) downBlock[0]);
            addChildBlock("down_" + ind + "_resnet2", (AbstractBlock) downBlock[1]);
            if (!isLast) {
                addChildBlock("down_" + ind + "_downsample", (AbstractBlock) downBlock[2]);
            }

            downs.add(downBlock);

            if (!isLast) {
                currentHorizon = currentHorizon / 2;
            }
        }

        // Middle blocks
        int midDim = dims[dims.length - 1];
        this.midBlock1 = new ResidualTemporalBlock(midDim, midDim, timeDim, currentHorizon);
        this.midBlock2 = new ResidualTemporalBlock(midDim, midDim, timeDim, currentHorizon);
        addChildBlock("midBlock1", midBlock1);
        addChildBlock("midBlock2", midBlock2);

        // Upsampling path - must have same number of upsamples as downsamples
        // to restore original horizon (e.g., 3 downsamples needs 3 upsamples)
        this.ups = new ArrayList<>();
        int numUpBlocks = numResolutions - 1;  // One less than down blocks (no up for last down)
        for (int ind = 0; ind < numUpBlocks; ind++) {
            int revInd = numUpBlocks - 1 - ind;
            int dimIn = inOut[revInd + 1][0];
            int dimOut = inOut[revInd + 1][1];

            Block[] upBlock = new Block[3];
            // Input channels are doubled due to skip connection
            upBlock[0] = new ResidualTemporalBlock(dimOut * 2, dimIn, timeDim, currentHorizon);
            upBlock[1] = new ResidualTemporalBlock(dimIn, dimIn, timeDim, currentHorizon);
            // All up blocks upsample to restore original horizon
            upBlock[2] = new Upsample1d(dimIn);

            addChildBlock("up_" + ind + "_resnet1", (AbstractBlock) upBlock[0]);
            addChildBlock("up_" + ind + "_resnet2", (AbstractBlock) upBlock[1]);
            addChildBlock("up_" + ind + "_upsample", (AbstractBlock) upBlock[2]);

            ups.add(upBlock);
            currentHorizon = currentHorizon * 2;
        }

        // Final convolution
        this.finalConv = new SequentialBlock();
        finalConv.add(new Conv1dBlock(dim, dim, 5));
        finalConv.add(Conv1d.builder()
                .setKernelShape(new Shape(1))
                .setFilters(transitionDim)
                .build());
        addChildBlock("finalConv", finalConv);
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs,
                                     boolean training, PairList<String, Object> params) {
        // inputs: [x, cond, time]
        // x: [batch, horizon, transition_dim]
        // cond: conditioning (unused in this implementation)
        // time: [batch]
        NDArray x = inputs.get(0);
        // NDArray cond = inputs.get(1);  // Currently unused
        NDArray time = inputs.get(2);

        // Rearrange x from [batch, horizon, transition] to [batch, transition, horizon]
        x = x.transpose(0, 2, 1);

        // Time embedding
        NDList tOut = timeMlp.forward(parameterStore, new NDList(time), training, params);
        NDArray t = tOut.singletonOrThrow();

        // Store skip connections
        List<NDArray> h = new ArrayList<>();

        // Downsampling path
        for (Block[] downBlock : downs) {
            ResidualTemporalBlock resnet1 = (ResidualTemporalBlock) downBlock[0];
            ResidualTemporalBlock resnet2 = (ResidualTemporalBlock) downBlock[1];
            Block downsample = downBlock[2];

            NDList res1Out = resnet1.forward(parameterStore, new NDList(x, t), training, params);
            x = res1Out.singletonOrThrow();

            NDList res2Out = resnet2.forward(parameterStore, new NDList(x, t), training, params);
            x = res2Out.singletonOrThrow();

            h.add(x);

            NDList downOut = downsample.forward(parameterStore, new NDList(x), training, params);
            x = downOut.singletonOrThrow();
        }

        // Middle blocks
        NDList mid1Out = midBlock1.forward(parameterStore, new NDList(x, t), training, params);
        x = mid1Out.singletonOrThrow();

        NDList mid2Out = midBlock2.forward(parameterStore, new NDList(x, t), training, params);
        x = mid2Out.singletonOrThrow();

        // Upsampling path
        for (Block[] upBlock : ups) {
            ResidualTemporalBlock resnet1 = (ResidualTemporalBlock) upBlock[0];
            ResidualTemporalBlock resnet2 = (ResidualTemporalBlock) upBlock[1];
            Block upsample = upBlock[2];

            // Concatenate skip connection
            NDArray skip = h.remove(h.size() - 1);
            x = x.concat(skip, 1);

            NDList res1Out = resnet1.forward(parameterStore, new NDList(x, t), training, params);
            x = res1Out.singletonOrThrow();

            NDList res2Out = resnet2.forward(parameterStore, new NDList(x, t), training, params);
            x = res2Out.singletonOrThrow();

            NDList upOut = upsample.forward(parameterStore, new NDList(x), training, params);
            x = upOut.singletonOrThrow();
        }

        // Final convolution
        NDList finalOut = finalConv.forward(parameterStore, new NDList(x), training, params);
        x = finalOut.singletonOrThrow();

        // Rearrange back to [batch, horizon, transition]
        x = x.transpose(0, 2, 1);

        return new NDList(x);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        // Output shape is same as input shape
        return new Shape[]{inputShapes[0]};
    }

    @Override
    protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        // inputShapes: [xShape, condShape, timeShape]
        // x: [batch, horizon, transitionDim]
        // cond: [batch, condDim]
        // time: [batch]
        Shape xShape = inputShapes[0];
        Shape timeShape = inputShapes[2];

        long batchSize = xShape.get(0);
        long horizon = xShape.get(1);

        // Initialize time MLP: input is [batch], output is [batch, dim]
        timeMlp.initialize(manager, dataType, timeShape);
        Shape tShape = new Shape(batchSize, dim);

        // Calculate channel dimensions for tracking shapes through the network
        int[] dims = new int[dimMults.length + 1];
        dims[0] = transitionDim;
        for (int i = 0; i < dimMults.length; i++) {
            dims[i + 1] = dim * dimMults[i];
        }

        // x is transposed to [batch, transitionDim, horizon] in forward pass
        long currentHorizon = horizon;
        int currentChannels = transitionDim;

        // Initialize downsampling blocks
        for (int ind = 0; ind < downs.size(); ind++) {
            Block[] downBlock = downs.get(ind);
            int dimIn = dims[ind];
            int dimOut = dims[ind + 1];
            boolean isLast = ind >= downs.size() - 1;

            ResidualTemporalBlock resnet1 = (ResidualTemporalBlock) downBlock[0];
            ResidualTemporalBlock resnet2 = (ResidualTemporalBlock) downBlock[1];
            Block downsample = downBlock[2];

            // resnet1: [batch, dimIn, horizon] + [batch, timeDim] -> [batch, dimOut, horizon]
            Shape xDownShape = new Shape(batchSize, dimIn, currentHorizon);
            resnet1.initialize(manager, dataType, xDownShape, tShape);

            // resnet2: [batch, dimOut, horizon] + [batch, timeDim] -> [batch, dimOut, horizon]
            Shape xAfterRes1 = new Shape(batchSize, dimOut, currentHorizon);
            resnet2.initialize(manager, dataType, xAfterRes1, tShape);

            // downsample: [batch, dimOut, horizon] -> [batch, dimOut, horizon/2]
            if (!isLast) {
                downsample.initialize(manager, dataType, xAfterRes1);
                currentHorizon = currentHorizon / 2;
            }

            currentChannels = dimOut;
        }

        // Initialize middle blocks
        int midDim = dims[dims.length - 1];
        Shape midShape = new Shape(batchSize, midDim, currentHorizon);
        midBlock1.initialize(manager, dataType, midShape, tShape);
        midBlock2.initialize(manager, dataType, midShape, tShape);

        // Initialize upsampling blocks - all blocks upsample to restore original horizon
        for (int ind = 0; ind < ups.size(); ind++) {
            Block[] upBlock = ups.get(ind);
            int revInd = ups.size() - 1 - ind;
            int dimIn = dims[revInd + 1];
            int dimOut = dims[revInd + 2];

            ResidualTemporalBlock resnet1 = (ResidualTemporalBlock) upBlock[0];
            ResidualTemporalBlock resnet2 = (ResidualTemporalBlock) upBlock[1];
            Block upsample = upBlock[2];

            // After concat with skip connection, channels are doubled
            Shape xUpShape = new Shape(batchSize, dimOut * 2, currentHorizon);
            resnet1.initialize(manager, dataType, xUpShape, tShape);

            Shape xAfterRes1 = new Shape(batchSize, dimIn, currentHorizon);
            resnet2.initialize(manager, dataType, xAfterRes1, tShape);

            // All up blocks upsample
            upsample.initialize(manager, dataType, xAfterRes1);
            currentHorizon = currentHorizon * 2;
        }

        // Initialize final conv: [batch, dim, horizon] -> [batch, transitionDim, horizon]
        Shape finalInShape = new Shape(batchSize, dim, currentHorizon);
        finalConv.initialize(manager, dataType, finalInShape);
    }

    @Override
    protected void saveMetadata(DataOutputStream os) throws IOException {
        os.writeInt(transitionDim);
        os.writeInt(dim);
        os.writeInt(dimMults.length);
        for (int m : dimMults) {
            os.writeInt(m);
        }
    }

    @Override
    public void loadMetadata(byte version, DataInputStream is)
            throws IOException, MalformedModelException {
        // Metadata is loaded via constructor, this is for compatibility
    }
}
