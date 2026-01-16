package com.technodrome.models.helpers;

import ai.djl.ndarray.NDArray;

import java.util.HashMap;
import java.util.Map;

/**
 * Weighted loss function for trajectory prediction.
 * Supports per-timestep and per-dimension weighting.
 */
public abstract class WeightedLoss {

    protected NDArray weights;
    protected final int actionDim;

    public WeightedLoss(NDArray weights, int actionDim) {
        this.weights = weights;
        this.actionDim = actionDim;
    }

    /**
     * Compute the element-wise loss (to be implemented by subclasses).
     */
    protected abstract NDArray computeLoss(NDArray pred, NDArray targ);

    /**
     * Compute weighted loss and info metrics.
     *
     * @param pred predictions [batch_size, horizon, transition_dim]
     * @param targ targets [batch_size, horizon, transition_dim]
     * @return LossResult containing weighted loss and info map
     */
    public LossResult forward(NDArray pred, NDArray targ) {
        // Compute element-wise loss
        NDArray loss = computeLoss(pred, targ);

        // Apply weights and compute mean
        NDArray weightedLoss = loss.mul(weights).mean();

        // Compute a0_loss (loss for first action, unweighted)
        NDArray a0Loss = loss.get(":, 0, :" + actionDim)
                .div(weights.get("0, :" + actionDim))
                .mean();

        Map<String, Float> info = new HashMap<>();
        info.put("a0_loss", a0Loss.getFloat());

        return new LossResult(weightedLoss, info);
    }

    /**
     * Result container for loss computation.
     */
    public static class LossResult {
        public final NDArray loss;
        public final Map<String, Float> info;

        public LossResult(NDArray loss, Map<String, Float> info) {
            this.loss = loss;
            this.info = info;
        }
    }

    /**
     * Weighted L1 loss.
     */
    public static class WeightedL1 extends WeightedLoss {
        public WeightedL1(NDArray weights, int actionDim) {
            super(weights, actionDim);
        }

        @Override
        protected NDArray computeLoss(NDArray pred, NDArray targ) {
            return pred.sub(targ).abs();
        }
    }

    /**
     * Weighted L2 loss (MSE).
     */
    public static class WeightedL2 extends WeightedLoss {
        public WeightedL2(NDArray weights, int actionDim) {
            super(weights, actionDim);
        }

        @Override
        protected NDArray computeLoss(NDArray pred, NDArray targ) {
            NDArray diff = pred.sub(targ);
            return diff.mul(diff);
        }
    }
}
