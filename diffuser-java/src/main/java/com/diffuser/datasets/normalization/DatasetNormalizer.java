package com.diffuser.datasets.normalization;

import java.util.HashMap;
import java.util.Map;

/**
 * Multi-field normalizer for datasets.
 * Maintains separate normalizers for different data fields (observations, actions, etc.).
 */
public class DatasetNormalizer {

    private final Map<String, Normalizer> normalizers;
    private final int observationDim;
    private final int actionDim;

    public DatasetNormalizer(Map<String, float[][]> dataset, String normalizerType, int[] pathLengths) {
        this.normalizers = new HashMap<>();

        // Flatten dataset
        Map<String, float[][]> flattenedDataset = flatten(dataset, pathLengths);

        // Get dimensions
        this.observationDim = flattenedDataset.get("observations")[0].length;
        this.actionDim = flattenedDataset.get("actions")[0].length;

        // Create normalizers for each field
        for (Map.Entry<String, float[][]> entry : flattenedDataset.entrySet()) {
            String key = entry.getKey();
            float[][] data = entry.getValue();

            try {
                Normalizer normalizer = createNormalizer(normalizerType, data);
                normalizers.put(key, normalizer);
            } catch (Exception e) {
                System.out.printf("[ DatasetNormalizer ] Skipping %s | %s%n", key, normalizerType);
            }
        }
    }

    private Normalizer createNormalizer(String type, float[][] data) {
        switch (type) {
            case "LimitsNormalizer":
                return new LimitsNormalizer(data);
            case "GaussianNormalizer":
                return new GaussianNormalizer(data);
            case "SafeLimitsNormalizer":
                return new SafeLimitsNormalizer(data);
            default:
                throw new IllegalArgumentException("Unknown normalizer type: " + type);
        }
    }

    /**
     * Flatten dataset from [n_episodes, max_path_length, dim] to [total_steps, dim].
     */
    private Map<String, float[][]> flatten(Map<String, float[][]> dataset, int[] pathLengths) {
        // For now, assume data is already in [total_steps, dim] format
        // In a full implementation, this would handle the episode structure
        return dataset;
    }

    /**
     * Normalize data for a given key.
     */
    public float[] normalize(float[] x, String key) {
        Normalizer normalizer = normalizers.get(key);
        if (normalizer == null) {
            throw new IllegalArgumentException("No normalizer found for key: " + key);
        }
        return normalizer.normalize(x);
    }

    /**
     * Unnormalize data for a given key.
     */
    public float[] unnormalize(float[] x, String key) {
        Normalizer normalizer = normalizers.get(key);
        if (normalizer == null) {
            throw new IllegalArgumentException("No normalizer found for key: " + key);
        }
        return normalizer.unnormalize(x);
    }

    /**
     * Normalize a batch of data.
     */
    public float[][] normalizeBatch(float[][] data, String key) {
        Normalizer normalizer = normalizers.get(key);
        if (normalizer == null) {
            throw new IllegalArgumentException("No normalizer found for key: " + key);
        }
        return normalizer.normalizeBatch(data);
    }

    /**
     * Unnormalize a batch of data.
     */
    public float[][] unnormalizeBatch(float[][] data, String key) {
        Normalizer normalizer = normalizers.get(key);
        if (normalizer == null) {
            throw new IllegalArgumentException("No normalizer found for key: " + key);
        }
        return normalizer.unnormalizeBatch(data);
    }

    public int getObservationDim() {
        return observationDim;
    }

    public int getActionDim() {
        return actionDim;
    }

    public Normalizer getNormalizer(String key) {
        return normalizers.get(key);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, Normalizer> entry : normalizers.entrySet()) {
            sb.append(String.format("%s: %s%n", entry.getKey(), entry.getValue()));
        }
        return sb.toString();
    }
}
