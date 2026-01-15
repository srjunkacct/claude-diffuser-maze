package com.diffuser.datasets.normalization;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class NormalizerTest {

    @Test
    void testLimitsNormalizerNormalize() {
        float[][] data = {
            {0, 0},
            {10, 20},
            {5, 10}
        };

        LimitsNormalizer normalizer = new LimitsNormalizer(data);

        // Test normalization
        float[] result = normalizer.normalize(new float[]{5, 10});

        // Expected: [0, 0] maps to [-1, -1], [10, 20] maps to [1, 1]
        // So [5, 10] should map to [0, 0]
        assertEquals(0.0f, result[0], 0.001f);
        assertEquals(0.0f, result[1], 0.001f);
    }

    @Test
    void testLimitsNormalizerUnnormalize() {
        float[][] data = {
            {0, 0},
            {10, 20}
        };

        LimitsNormalizer normalizer = new LimitsNormalizer(data);

        // Test unnormalization
        float[] result = normalizer.unnormalize(new float[]{0, 0});

        // [0, 0] in normalized space should map back to midpoint
        assertEquals(5.0f, result[0], 0.001f);
        assertEquals(10.0f, result[1], 0.001f);
    }

    @Test
    void testLimitsNormalizerRoundTrip() {
        float[][] data = {
            {-5, 0},
            {15, 100}
        };

        LimitsNormalizer normalizer = new LimitsNormalizer(data);

        float[] original = {7.5f, 50.0f};
        float[] normalized = normalizer.normalize(original);
        float[] recovered = normalizer.unnormalize(normalized);

        assertArrayEquals(original, recovered, 0.001f);
    }

    @Test
    void testGaussianNormalizerNormalize() {
        // Create data with known mean and std
        float[][] data = {
            {0, 0},
            {2, 4},
            {4, 8}
        };
        // Mean: [2, 4], Std: [1.63, 3.27]

        GaussianNormalizer normalizer = new GaussianNormalizer(data);

        // Test that mean normalizes to 0
        float[] meanNorm = normalizer.normalize(new float[]{2, 4});
        assertEquals(0.0f, meanNorm[0], 0.001f);
        assertEquals(0.0f, meanNorm[1], 0.001f);
    }

    @Test
    void testGaussianNormalizerRoundTrip() {
        float[][] data = {
            {-10, 5},
            {0, 15},
            {10, 25}
        };

        GaussianNormalizer normalizer = new GaussianNormalizer(data);

        float[] original = {5.0f, 20.0f};
        float[] normalized = normalizer.normalize(original);
        float[] recovered = normalizer.unnormalize(normalized);

        assertArrayEquals(original, recovered, 0.01f);
    }

    @Test
    void testSafeLimitsNormalizerHandlesConstant() {
        // Data with constant dimension
        float[][] data = {
            {0, 5},
            {10, 5},
            {5, 5}
        };

        // Should not throw exception
        SafeLimitsNormalizer normalizer = new SafeLimitsNormalizer(data);

        // Should be able to normalize without NaN
        float[] result = normalizer.normalize(new float[]{5, 5});
        assertFalse(Float.isNaN(result[0]));
        assertFalse(Float.isNaN(result[1]));
    }
}
