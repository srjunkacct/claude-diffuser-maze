package com.technodrome.models.helpers;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.util.Map;

/**
 * Helper functions for diffusion models.
 */
public final class DiffusionHelpers {

    private DiffusionHelpers() {
        // Utility class
    }

    /**
     * Extract values from array a at indices t, and reshape for broadcasting.
     *
     * @param a       1D array of values indexed by timestep
     * @param t       batch of timestep indices [batch_size]
     * @param xShape  target shape for broadcasting
     * @return extracted values reshaped for broadcasting
     */
    public static NDArray extract(NDArray a, NDArray t, Shape xShape) {
        // Gather values: out = a[t]
        NDArray out = a.get(t);

        // Reshape to [batch, 1, 1, ...] for broadcasting with xShape
        long batchSize = t.getShape().get(0);
        long[] newShape = new long[xShape.dimension()];
        newShape[0] = batchSize;
        for (int i = 1; i < newShape.length; i++) {
            newShape[i] = 1;
        }

        return out.reshape(newShape);
    }

    /**
     * Cosine beta schedule for diffusion process.
     * As proposed in https://openreview.net/forum?id=-NEXDKk8gZ
     *
     * @param manager    NDManager for array creation
     * @param timesteps  number of diffusion timesteps
     * @param s          offset parameter (default 0.008)
     * @return betas array of shape [timesteps]
     */
    public static NDArray cosineBetaSchedule(NDManager manager, int timesteps, double s) {
        int steps = timesteps + 1;
        NDArray x = manager.linspace(0, steps, steps).toType(DataType.FLOAT64, false);

        // alphas_cumprod = cos(((x / steps) + s) / (1 + s) * pi * 0.5) ** 2
        NDArray alphasCumprod = x.div(steps).add(s).div(1 + s).mul(Math.PI * 0.5);
        alphasCumprod = alphasCumprod.cos().pow(2);

        // Normalize: alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        double firstVal = alphasCumprod.getDouble(0);
        alphasCumprod = alphasCumprod.div(firstVal);

        // betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        NDArray alphasCumprodNext = alphasCumprod.get(new NDIndex("1:"));
        NDArray alphasCumprodPrev = alphasCumprod.get(new NDIndex(":-1"));
        NDArray betas = alphasCumprodNext.div(alphasCumprodPrev).neg().add(1);

        // Clip betas to [0, 0.999]
        betas = betas.clip(0, 0.999);

        return betas.toType(DataType.FLOAT32, false);
    }

    /**
     * Cosine beta schedule with default s=0.008.
     */
    public static NDArray cosineBetaSchedule(NDManager manager, int timesteps) {
        return cosineBetaSchedule(manager, timesteps, 0.008);
    }

    /**
     * Apply conditioning to trajectory.
     * Sets observation values at specified timesteps.
     *
     * @param x          trajectory tensor [batch, horizon, transition_dim]
     * @param conditions map of timestep -> observation values
     * @param actionDim  dimension of action space
     * @return conditioned trajectory
     */
    public static NDArray applyConditioning(NDArray x, Map<Integer, NDArray> conditions, int actionDim) {
        NDArray result = x.duplicate();
        for (Map.Entry<Integer, NDArray> entry : conditions.entrySet()) {
            int t = entry.getKey();
            NDArray val = entry.getValue();
            // Set x[:, t, actionDim:] = val
            // This sets the observation part at timestep t
            result.set(new NDIndex(":, {}, {}:", t, actionDim), val);
        }
        return result;
    }
}
