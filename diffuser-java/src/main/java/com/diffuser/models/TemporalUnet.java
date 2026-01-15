package com.diffuser.models;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv1d;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import com.diffuser.models.helpers.Conv1dBlock;
import com.diffuser.models.helpers.Downsample1d;
import com.diffuser.models.helpers.Mish;
import com.diffuser.models.helpers.SinusoidalPosEmb;
import com.diffuser.models.helpers.Upsample1d;

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

        // Upsampling path
        this.ups = new ArrayList<>();
        for (int ind = 0; ind < numResolutions - 1; ind++) {
            int revInd = numResolutions - 2 - ind;
            int dimIn = inOut[revInd + 1][0];
            int dimOut = inOut[revInd + 1][1];
            boolean isLast = ind >= numResolutions - 2;

            Block[] upBlock = new Block[3];
            // Input channels are doubled due to skip connection
            upBlock[0] = new ResidualTemporalBlock(dimOut * 2, dimIn, timeDim, currentHorizon);
            upBlock[1] = new ResidualTemporalBlock(dimIn, dimIn, timeDim, currentHorizon);
            upBlock[2] = isLast ? Blocks.identityBlock() : new Upsample1d(dimIn);

            addChildBlock("up_" + ind + "_resnet1", (AbstractBlock) upBlock[0]);
            addChildBlock("up_" + ind + "_resnet2", (AbstractBlock) upBlock[1]);
            if (!isLast) {
                addChildBlock("up_" + ind + "_upsample", (AbstractBlock) upBlock[2]);
            }

            ups.add(upBlock);

            if (!isLast) {
                currentHorizon = currentHorizon * 2;
            }
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
}
