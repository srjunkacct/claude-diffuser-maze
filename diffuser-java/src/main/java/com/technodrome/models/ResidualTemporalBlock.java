package com.technodrome.models;

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

import com.technodrome.models.helpers.Conv1dBlock;
import com.technodrome.models.helpers.Mish;

/**
 * Residual Temporal Block for processing temporal sequences.
 * Combines two Conv1d blocks with time embedding injection and a residual connection.
 */
public class ResidualTemporalBlock extends AbstractBlock {

    private static final byte VERSION = 1;

    private final Conv1dBlock block1;
    private final Conv1dBlock block2;
    private final SequentialBlock timeMlp;
    private final Block residualConv;

    public ResidualTemporalBlock(int inChannels, int outChannels, int embedDim, int horizon) {
        this(inChannels, outChannels, embedDim, horizon, 5);
    }

    public ResidualTemporalBlock(int inChannels, int outChannels, int embedDim, int horizon, int kernelSize) {
        super(VERSION);

        this.block1 = new Conv1dBlock(inChannels, outChannels, kernelSize);
        addChildBlock("block1", block1);

        this.block2 = new Conv1dBlock(outChannels, outChannels, kernelSize);
        addChildBlock("block2", block2);

        // Time MLP: Mish -> Linear -> Rearrange (reshape in forward)
        this.timeMlp = new SequentialBlock();
        timeMlp.add(new Mish());
        timeMlp.add(Linear.builder().setUnits(outChannels).build());
        addChildBlock("timeMlp", timeMlp);

        // Residual connection: 1x1 conv if channels differ, otherwise identity
        if (inChannels != outChannels) {
            this.residualConv = Conv1d.builder()
                    .setKernelShape(new Shape(1))
                    .setFilters(outChannels)
                    .build();
            addChildBlock("residualConv", (AbstractBlock) residualConv);
        } else {
            this.residualConv = Blocks.identityBlock();
        }
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs,
                                     boolean training, PairList<String, Object> params) {
        // inputs: [x, t] where x is [batch, channels, horizon] and t is [batch, embedDim]
        NDArray x = inputs.get(0);
        NDArray t = inputs.get(1);

        // First conv block
        NDList block1Out = block1.forward(parameterStore, new NDList(x), training, params);
        NDArray out = block1Out.singletonOrThrow();

        // Time embedding: timeMlp(t) -> reshape to [batch, outChannels, 1] for broadcasting
        NDList tOut = timeMlp.forward(parameterStore, new NDList(t), training, params);
        NDArray tEmb = tOut.singletonOrThrow();
        tEmb = tEmb.expandDims(-1);  // [batch, outChannels, 1]

        // Add time embedding
        out = out.add(tEmb);

        // Second conv block
        NDList block2Out = block2.forward(parameterStore, new NDList(out), training, params);
        out = block2Out.singletonOrThrow();

        // Residual connection
        NDList residualOut = residualConv.forward(parameterStore, new NDList(x), training, params);
        NDArray residual = residualOut.singletonOrThrow();

        return new NDList(out.add(residual));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        // Output shape matches first conv block output
        return block1.getOutputShapes(new Shape[]{inputShapes[0]});
    }
}
