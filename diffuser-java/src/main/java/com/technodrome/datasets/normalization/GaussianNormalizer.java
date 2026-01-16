package com.technodrome.datasets.normalization;

/**
 * Normalizer that transforms data to zero mean and unit variance.
 */
public class GaussianNormalizer extends Normalizer {

    private final float[] means;
    private final float[] stds;

    public GaussianNormalizer(float[][] data) {
        super(data);

        this.means = new float[dim];
        this.stds = new float[dim];

        int n = data.length;

        // Compute means
        for (float[] row : data) {
            for (int d = 0; d < dim; d++) {
                means[d] += row[d];
            }
        }
        for (int d = 0; d < dim; d++) {
            means[d] /= n;
        }

        // Compute standard deviations
        for (float[] row : data) {
            for (int d = 0; d < dim; d++) {
                float diff = row[d] - means[d];
                stds[d] += diff * diff;
            }
        }
        for (int d = 0; d < dim; d++) {
            stds[d] = (float) Math.sqrt(stds[d] / n);
            // Prevent division by zero
            if (stds[d] < 1e-8) {
                stds[d] = 1.0f;
            }
        }
    }

    @Override
    public float[] normalize(float[] x) {
        float[] result = new float[dim];
        for (int d = 0; d < dim; d++) {
            result[d] = (x[d] - means[d]) / stds[d];
        }
        return result;
    }

    @Override
    public float[] unnormalize(float[] x) {
        float[] result = new float[dim];
        for (int d = 0; d < dim; d++) {
            result[d] = x[d] * stds[d] + means[d];
        }
        return result;
    }

    public float[] getMeans() {
        return means.clone();
    }

    public float[] getStds() {
        return stds.clone();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("[ GaussianNormalizer ] dim: %d%n", dim));
        sb.append("    means: [");
        for (int i = 0; i < Math.min(dim, 5); i++) {
            sb.append(String.format("%.2f", means[i]));
            if (i < Math.min(dim, 5) - 1) sb.append(", ");
        }
        if (dim > 5) sb.append(", ...");
        sb.append("]\n");
        sb.append("    stds: [");
        for (int i = 0; i < Math.min(dim, 5); i++) {
            sb.append(String.format("%.2f", stds[i]));
            if (i < Math.min(dim, 5) - 1) sb.append(", ");
        }
        if (dim > 5) sb.append(", ...");
        sb.append("]");
        return sb.toString();
    }
}
