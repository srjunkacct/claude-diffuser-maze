package com.technodrome.models.helpers;

import ai.djl.ndarray.NDArray;

import java.util.HashMap;
import java.util.Map;

/**
 * Value loss function for value function training.
 * Includes correlation metrics for monitoring.
 */
public abstract class ValueLoss {

    public ValueLoss() {
    }

    /**
     * Compute the element-wise loss (to be implemented by subclasses).
     */
    protected abstract NDArray computeLoss(NDArray pred, NDArray targ);

    /**
     * Compute loss and info metrics for value function.
     *
     * @param pred predictions [batch_size, 1]
     * @param targ targets [batch_size, 1]
     * @return LossResult containing loss and info map
     */
    public LossResult forward(NDArray pred, NDArray targ) {
        NDArray loss = computeLoss(pred, targ).mean();

        Map<String, Float> info = new HashMap<>();
        info.put("mean_pred", pred.mean().getFloat());
        info.put("mean_targ", targ.mean().getFloat());
        info.put("min_pred", pred.min().getFloat());
        info.put("min_targ", targ.min().getFloat());
        info.put("max_pred", pred.max().getFloat());
        info.put("max_targ", targ.max().getFloat());

        // Compute correlation if batch size > 1
        if (pred.size() > 1) {
            float corr = computeCorrelation(pred.flatten(), targ.flatten());
            info.put("corr", corr);
        } else {
            info.put("corr", Float.NaN);
        }

        return new LossResult(loss, info);
    }

    /**
     * Compute Pearson correlation coefficient.
     */
    private float computeCorrelation(NDArray x, NDArray y) {
        NDArray xMean = x.mean();
        NDArray yMean = y.mean();

        NDArray xCentered = x.sub(xMean);
        NDArray yCentered = y.sub(yMean);

        NDArray numerator = xCentered.mul(yCentered).sum();
        NDArray denomX = xCentered.mul(xCentered).sum().sqrt();
        NDArray denomY = yCentered.mul(yCentered).sum().sqrt();

        NDArray denom = denomX.mul(denomY);
        if (denom.getFloat() < 1e-8) {
            return Float.NaN;
        }

        return numerator.div(denom).getFloat();
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
     * Value L1 loss.
     */
    public static class ValueL1 extends ValueLoss {
        @Override
        protected NDArray computeLoss(NDArray pred, NDArray targ) {
            return pred.sub(targ).abs();
        }
    }

    /**
     * Value L2 loss (MSE).
     */
    public static class ValueL2 extends ValueLoss {
        @Override
        protected NDArray computeLoss(NDArray pred, NDArray targ) {
            NDArray diff = pred.sub(targ);
            return diff.mul(diff);
        }
    }
}
