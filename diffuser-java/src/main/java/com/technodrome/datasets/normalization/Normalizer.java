package com.technodrome.datasets.normalization;

/**
 * Base class for data normalizers.
 * Subclasses implement specific normalization strategies.
 */
public abstract class Normalizer {

    protected final float[] mins;
    protected final float[] maxs;
    protected final int dim;

    public Normalizer(float[][] data) {
        this.dim = data[0].length;
        this.mins = new float[dim];
        this.maxs = new float[dim];

        // Initialize mins and maxs
        for (int d = 0; d < dim; d++) {
            mins[d] = Float.MAX_VALUE;
            maxs[d] = Float.MIN_VALUE;
        }

        // Compute mins and maxs
        for (float[] row : data) {
            for (int d = 0; d < dim; d++) {
                mins[d] = Math.min(mins[d], row[d]);
                maxs[d] = Math.max(maxs[d], row[d]);
            }
        }
    }

    /**
     * Normalize a single data point.
     */
    public abstract float[] normalize(float[] x);

    /**
     * Unnormalize a single data point.
     */
    public abstract float[] unnormalize(float[] x);

    /**
     * Normalize a batch of data points.
     */
    public float[][] normalizeBatch(float[][] data) {
        float[][] result = new float[data.length][];
        for (int i = 0; i < data.length; i++) {
            result[i] = normalize(data[i]);
        }
        return result;
    }

    /**
     * Unnormalize a batch of data points.
     */
    public float[][] unnormalizeBatch(float[][] data) {
        float[][] result = new float[data.length][];
        for (int i = 0; i < data.length; i++) {
            result[i] = unnormalize(data[i]);
        }
        return result;
    }

    public int getDim() {
        return dim;
    }

    public float[] getMins() {
        return mins.clone();
    }

    public float[] getMaxs() {
        return maxs.clone();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("[ Normalizer ] dim: %d%n", dim));
        sb.append("    -: [");
        for (int i = 0; i < Math.min(dim, 5); i++) {
            sb.append(String.format("%.2f", mins[i]));
            if (i < Math.min(dim, 5) - 1) sb.append(", ");
        }
        if (dim > 5) sb.append(", ...");
        sb.append("]\n");
        sb.append("    +: [");
        for (int i = 0; i < Math.min(dim, 5); i++) {
            sb.append(String.format("%.2f", maxs[i]));
            if (i < Math.min(dim, 5) - 1) sb.append(", ");
        }
        if (dim > 5) sb.append(", ...");
        sb.append("]");
        return sb.toString();
    }
}
