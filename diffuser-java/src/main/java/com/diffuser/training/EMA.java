package com.diffuser.training;

import ai.djl.ndarray.NDArray;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;

/**
 * Exponential Moving Average for model parameters.
 * Maintains a slow-moving copy of model parameters for stable inference.
 */
public class EMA {

    private final float beta;

    public EMA(float beta) {
        this.beta = beta;
    }

    /**
     * Update moving average model parameters from current model.
     *
     * @param maModel      the moving average model to update
     * @param currentModel the current (training) model
     */
    public void updateModelAverage(Block maModel, Block currentModel) {
        // Iterate through parameters using the ParameterList
        var maParams = maModel.getParameters();
        var currentParams = currentModel.getParameters();

        // Both parameter lists should have the same size and order
        for (int i = 0; i < currentParams.size(); i++) {
            Parameter currentParam = currentParams.valueAt(i);
            Parameter maParam = maParams.valueAt(i);

            if (maParam != null && currentParam != null &&
                currentParam.getArray() != null && maParam.getArray() != null) {
                NDArray oldWeight = maParam.getArray();
                NDArray newWeight = currentParam.getArray();
                NDArray updatedWeight = updateAverage(oldWeight, newWeight);
                maParam.setArray(updatedWeight);
            }
        }
    }

    /**
     * Compute updated average: old * beta + new * (1 - beta)
     */
    private NDArray updateAverage(NDArray old, NDArray current) {
        return old.mul(beta).add(current.mul(1 - beta));
    }

    public float getBeta() {
        return beta;
    }
}
