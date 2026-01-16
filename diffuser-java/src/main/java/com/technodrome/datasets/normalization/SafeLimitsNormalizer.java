package com.technodrome.datasets.normalization;

/**
 * Normalizer that handles constant dimensions safely.
 * Extends LimitsNormalizer with handling for dimensions where min == max.
 */
public class SafeLimitsNormalizer extends LimitsNormalizer {

    private static final float DEFAULT_EPS = 1.0f;

    public SafeLimitsNormalizer(float[][] data) {
        this(data, DEFAULT_EPS);
    }

    public SafeLimitsNormalizer(float[][] data, float eps) {
        super(data);

        // Handle constant dimensions
        for (int d = 0; d < dim; d++) {
            if (Math.abs(mins[d] - maxs[d]) < 1e-8) {
                System.out.printf("[ SafeLimitsNormalizer ] Constant data in dimension %d | max = min = %.4f%n",
                        d, maxs[d]);
                mins[d] -= eps;
                maxs[d] += eps;
            }
        }
    }
}
