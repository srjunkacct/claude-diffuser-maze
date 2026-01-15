package com.diffuser.datasets.normalization;

/**
 * Normalizer that maps data to [-1, 1] range based on min/max values.
 */
public class LimitsNormalizer extends Normalizer {

    public LimitsNormalizer(float[][] data) {
        super(data);
    }

    @Override
    public float[] normalize(float[] x) {
        float[] result = new float[dim];
        for (int d = 0; d < dim; d++) {
            float range = maxs[d] - mins[d];
            if (range < 1e-8) {
                result[d] = 0;  // Handle constant dimensions
            } else {
                // Map to [0, 1]
                result[d] = (x[d] - mins[d]) / range;
                // Map to [-1, 1]
                result[d] = 2 * result[d] - 1;
            }
        }
        return result;
    }

    @Override
    public float[] unnormalize(float[] x) {
        float[] result = new float[dim];
        float eps = 1e-4f;

        for (int d = 0; d < dim; d++) {
            float val = x[d];

            // Clip to [-1, 1] with warning
            if (val > 1 + eps || val < -1 - eps) {
                val = Math.max(-1, Math.min(1, val));
            }

            // Map from [-1, 1] to [0, 1]
            val = (val + 1) / 2.0f;

            // Map to original range
            result[d] = val * (maxs[d] - mins[d]) + mins[d];
        }
        return result;
    }
}
