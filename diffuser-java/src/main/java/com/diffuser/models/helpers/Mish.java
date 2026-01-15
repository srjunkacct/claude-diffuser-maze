package com.diffuser.models.helpers;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * Mish activation function: x * tanh(softplus(x))
 * where softplus(x) = ln(1 + e^x)
 */
public class Mish extends AbstractBlock {

    private static final byte VERSION = 1;

    public Mish() {
        super(VERSION);
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs,
                                     boolean training, PairList<String, Object> params) {
        NDArray x = inputs.singletonOrThrow();
        // Mish: x * tanh(softplus(x))
        // softplus(x) = log(1 + exp(x))
        NDArray softplus = x.exp().add(1).log();
        NDArray result = x.mul(softplus.tanh());
        return new NDList(result);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return inputShapes;
    }
}
