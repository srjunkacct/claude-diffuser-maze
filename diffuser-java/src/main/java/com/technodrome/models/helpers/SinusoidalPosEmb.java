package com.technodrome.models.helpers;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * Sinusoidal Positional Embedding for diffusion timestep encoding.
 * Maps scalar timesteps to high-dimensional embeddings using sine and cosine functions.
 */
public class SinusoidalPosEmb extends AbstractBlock {

    private static final byte VERSION = 1;
    private final int dim;

    public SinusoidalPosEmb(int dim) {
        super(VERSION);
        this.dim = dim;
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs,
                                     boolean training, PairList<String, Object> params) {
        NDArray x = inputs.singletonOrThrow();
        NDManager manager = x.getManager();

        int halfDim = dim / 2;
        double embScale = Math.log(10000.0) / (halfDim - 1);

        // Create embedding scale: exp(arange(halfDim) * -embScale)
        NDArray embIndices = manager.arange(halfDim).toType(DataType.FLOAT32, false);
        NDArray emb = embIndices.mul(-embScale).exp();

        // x[:, None] * emb[None, :]
        NDArray xExpanded = x.expandDims(-1);  // [batch, 1]
        NDArray embExpanded = emb.expandDims(0);  // [1, halfDim]
        NDArray embProduct = xExpanded.mul(embExpanded);  // [batch, halfDim]

        // Concatenate sin and cos
        NDArray sinEmb = embProduct.sin();
        NDArray cosEmb = embProduct.cos();

        return new NDList(sinEmb.concat(cosEmb, -1));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[]{new Shape(inputShapes[0].get(0), dim)};
    }
}
