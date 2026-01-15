package com.diffuser.models.helpers;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.convolutional.Conv1dTranspose;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * 1D Upsampling layer using transposed convolution.
 * Increases the temporal dimension by factor of 2.
 */
public class Upsample1d extends AbstractBlock {

    private static final byte VERSION = 1;
    private final Conv1dTranspose conv;

    public Upsample1d(int dim) {
        super(VERSION);
        // ConvTranspose1d with kernel=4, stride=2, padding=1
        this.conv = Conv1dTranspose.builder()
                .setKernelShape(new Shape(4))
                .setFilters(dim)
                .optStride(new Shape(2))
                .optPadding(new Shape(1))
                .build();
        addChildBlock("conv", conv);
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs,
                                     boolean training, PairList<String, Object> params) {
        return conv.forward(parameterStore, inputs, training, params);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return conv.getOutputShapes(inputShapes);
    }
}
