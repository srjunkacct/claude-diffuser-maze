package com.technodrome.models.helpers;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.convolutional.Conv1d;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * 1D Upsampling layer using nearest-neighbor interpolation followed by Conv1d.
 * Increases the temporal dimension by factor of 2.
 *
 * Note: Uses interpolation + Conv1d instead of Conv1dTranspose to avoid
 * a DJL bug where Conv1dTranspose incorrectly calls randomFlipTopBottom.
 */
public class Upsample1d extends AbstractBlock {

    private static final byte VERSION = 1;
    private final int dim;
    private final Conv1d conv;

    public Upsample1d(int dim) {
        super(VERSION);
        this.dim = dim;

        // Conv1d with kernel=3, padding=1 to refine after upsampling
        this.conv = Conv1d.builder()
                .setKernelShape(new Shape(3))
                .setFilters(dim)
                .optPadding(new Shape(1))
                .build();
        addChildBlock("conv", conv);
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs,
                                     boolean training, PairList<String, Object> params) {
        NDArray x = inputs.singletonOrThrow();

        // x shape: [batch, channels, length]
        // Upsample by factor of 2 using nearest-neighbor interpolation
        x = upsampleNearest(x, 2);

        // Apply conv to refine
        return conv.forward(parameterStore, new NDList(x), training, params);
    }

    /**
     * Nearest-neighbor upsampling for 1D data.
     * Each element is repeated 'scale' times along the last dimension.
     */
    private NDArray upsampleNearest(NDArray x, int scale) {
        // x: [batch, channels, length]
        Shape shape = x.getShape();
        long batch = shape.get(0);
        long channels = shape.get(1);
        long length = shape.get(2);

        // Expand and repeat along new dimension
        x = x.expandDims(-1);           // [batch, channels, length, 1]
        x = x.repeat(3, scale);         // [batch, channels, length, scale]

        // Reshape to interleave: [batch, channels, length * scale]
        return x.reshape(batch, channels, length * scale);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        // Input: [batch, channels, length]
        // Output: [batch, dim, length * 2]
        Shape input = inputShapes[0];
        return new Shape[]{new Shape(input.get(0), dim, input.get(2) * 2)};
    }

    @Override
    protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        // After upsampling, shape is [batch, channels, length * 2]
        Shape inputShape = inputShapes[0];
        Shape upsampledShape = new Shape(inputShape.get(0), inputShape.get(1), inputShape.get(2) * 2);
        conv.initialize(manager, dataType, upsampledShape);
    }
}
