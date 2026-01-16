package com.technodrome.models.helpers;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.convolutional.Conv1d;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * Conv1d -> BatchNorm -> Mish activation block.
 * Standard building block for temporal processing.
 * Note: Uses BatchNorm instead of GroupNorm as GroupNorm is not directly available in DJL.
 */
public class Conv1dBlock extends AbstractBlock {

    private static final byte VERSION = 1;
    private final Conv1d conv;
    private final BatchNorm batchNorm;
    private final Mish mish;

    public Conv1dBlock(int inChannels, int outChannels, int kernelSize) {
        this(inChannels, outChannels, kernelSize, 8);
    }

    public Conv1dBlock(int inChannels, int outChannels, int kernelSize, int nGroups) {
        super(VERSION);

        int padding = kernelSize / 2;

        this.conv = Conv1d.builder()
                .setKernelShape(new Shape(kernelSize))
                .setFilters(outChannels)
                .optPadding(new Shape(padding))
                .build();
        addChildBlock("conv", conv);

        // Using BatchNorm instead of GroupNorm
        this.batchNorm = BatchNorm.builder().optAxis(1).build();
        addChildBlock("batchNorm", batchNorm);

        this.mish = new Mish();
        addChildBlock("mish", mish);
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs,
                                     boolean training, PairList<String, Object> params) {
        NDArray x = inputs.singletonOrThrow();

        // Conv1d
        NDList convOut = conv.forward(parameterStore, new NDList(x), training, params);
        x = convOut.singletonOrThrow();

        // BatchNorm
        NDList normOut = batchNorm.forward(parameterStore, new NDList(x), training, params);
        x = normOut.singletonOrThrow();

        // Mish activation
        NDList mishOut = mish.forward(parameterStore, new NDList(x), training, params);

        return mishOut;
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        Shape[] convShapes = conv.getOutputShapes(inputShapes);
        return convShapes;  // BatchNorm and Mish preserve shape
    }
}
