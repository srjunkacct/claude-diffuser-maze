package com.diffuser.models;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import com.diffuser.models.helpers.Downsample1d;
import com.diffuser.models.helpers.Mish;
import com.diffuser.models.helpers.SinusoidalPosEmb;

import java.util.ArrayList;
import java.util.List;

/**
 * Temporal Value network for estimating trajectory values.
 * Similar architecture to TemporalUnet but outputs scalar values.
 */
public class TemporalValue extends AbstractBlock {

    private static final byte VERSION = 1;

    private final int transitionDim;
    private final int dim;
    private final int outDim;
    private final int[] dimMults;

    private final SequentialBlock timeMlp;
    private final List<TemporalBlock> blocks;
    private final SequentialBlock finalBlock;
    private final int fcDim;
    private final int timeDim;

    /**
     * Container for a temporal block (two resnets + downsample).
     */
    private static class TemporalBlock {
        final ResidualTemporalBlock resnet1;
        final ResidualTemporalBlock resnet2;
        final Downsample1d downsample;

        TemporalBlock(ResidualTemporalBlock resnet1, ResidualTemporalBlock resnet2, Downsample1d downsample) {
            this.resnet1 = resnet1;
            this.resnet2 = resnet2;
            this.downsample = downsample;
        }
    }

    public TemporalValue(int horizon, int transitionDim, int condDim) {
        this(horizon, transitionDim, condDim, 32, null, 1, new int[]{1, 2, 4, 8});
    }

    public TemporalValue(int horizon, int transitionDim, int condDim, int dim,
                         Integer timeDimOverride, int outDim, int[] dimMults) {
        super(VERSION);

        this.transitionDim = transitionDim;
        this.dim = dim;
        this.outDim = outDim;
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

        System.out.println("[ models/temporal ] TemporalValue channel dimensions:");
        for (int[] pair : inOut) {
            System.out.println("  " + pair[0] + " -> " + pair[1]);
        }

        this.timeDim = timeDimOverride != null ? timeDimOverride : dim;

        // Time MLP: SinusoidalPosEmb -> Linear -> Mish -> Linear
        this.timeMlp = new SequentialBlock();
        timeMlp.add(new SinusoidalPosEmb(dim));
        timeMlp.add(Linear.builder().setUnits(dim * 4).build());
        timeMlp.add(new Mish());
        timeMlp.add(Linear.builder().setUnits(dim).build());
        addChildBlock("timeMlp", timeMlp);

        // Downsampling blocks
        this.blocks = new ArrayList<>();
        int currentHorizon = horizon;

        for (int ind = 0; ind < inOut.length; ind++) {
            int dimIn = inOut[ind][0];
            int dimOut = inOut[ind][1];

            ResidualTemporalBlock resnet1 = new ResidualTemporalBlock(dimIn, dimOut, timeDim, currentHorizon, 5);
            ResidualTemporalBlock resnet2 = new ResidualTemporalBlock(dimOut, dimOut, timeDim, currentHorizon, 5);
            Downsample1d downsample = new Downsample1d(dimOut);

            addChildBlock("block_" + ind + "_resnet1", resnet1);
            addChildBlock("block_" + ind + "_resnet2", resnet2);
            addChildBlock("block_" + ind + "_downsample", downsample);

            blocks.add(new TemporalBlock(resnet1, resnet2, downsample));

            currentHorizon = currentHorizon / 2;
        }

        // Calculate final FC dimension
        this.fcDim = dims[dims.length - 1] * Math.max(currentHorizon, 1);

        // Final MLP block
        this.finalBlock = new SequentialBlock();
        finalBlock.add(Linear.builder().setUnits(fcDim / 2).build());
        finalBlock.add(new Mish());
        finalBlock.add(Linear.builder().setUnits(outDim).build());
        addChildBlock("finalBlock", finalBlock);
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs,
                                     boolean training, PairList<String, Object> params) {
        // inputs: [x, cond, time]
        // x: [batch, horizon, transition_dim]
        NDArray x = inputs.get(0);
        // NDArray cond = inputs.get(1);  // Currently unused
        NDArray time = inputs.get(2);

        // Rearrange x from [batch, horizon, transition] to [batch, transition, horizon]
        x = x.transpose(0, 2, 1);

        // Time embedding
        NDList tOut = timeMlp.forward(parameterStore, new NDList(time), training, params);
        NDArray t = tOut.singletonOrThrow();

        // Downsampling blocks
        for (TemporalBlock block : blocks) {
            NDList res1Out = block.resnet1.forward(parameterStore, new NDList(x, t), training, params);
            x = res1Out.singletonOrThrow();

            NDList res2Out = block.resnet2.forward(parameterStore, new NDList(x, t), training, params);
            x = res2Out.singletonOrThrow();

            NDList downOut = block.downsample.forward(parameterStore, new NDList(x), training, params);
            x = downOut.singletonOrThrow();
        }

        // Flatten and concatenate with time embedding
        long batchSize = x.getShape().get(0);
        x = x.reshape(batchSize, -1);
        x = x.concat(t, -1);

        // Final MLP
        NDList finalOut = finalBlock.forward(parameterStore, new NDList(x), training, params);

        return finalOut;
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[]{new Shape(inputShapes[0].get(0), outDim)};
    }
}
