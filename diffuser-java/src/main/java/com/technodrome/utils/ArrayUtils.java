package com.technodrome.utils;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

/**
 * Utility functions for array operations and conversions.
 */
public final class ArrayUtils {

    public static final DataType DEFAULT_DTYPE = DataType.FLOAT32;
    public static final Device DEFAULT_DEVICE = Device.gpu();

    private ArrayUtils() {
        // Utility class
    }

    /**
     * Convert NDArray to float array.
     */
    public static float[] toFloatArray(NDArray x) {
        return x.toFloatArray();
    }

    /**
     * Convert NDArray to 2D float array.
     */
    public static float[][] toFloatArray2D(NDArray x) {
        long[] shape = x.getShape().getShape();
        if (shape.length != 2) {
            throw new IllegalArgumentException("Expected 2D array, got " + shape.length + "D");
        }
        int rows = (int) shape[0];
        int cols = (int) shape[1];
        float[] flat = x.toFloatArray();
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(flat, i * cols, result[i], 0, cols);
        }
        return result;
    }

    /**
     * Convert float array to NDArray.
     */
    public static NDArray toNDArray(NDManager manager, float[] x) {
        return manager.create(x);
    }

    /**
     * Convert 2D float array to NDArray.
     */
    public static NDArray toNDArray(NDManager manager, float[][] x) {
        return manager.create(x);
    }

    /**
     * Move NDArray to specified device.
     */
    public static NDArray toDevice(NDArray x, Device device) {
        return x.toDevice(device, false);
    }

    /**
     * Move map of NDArrays to specified device.
     */
    public static Map<Integer, NDArray> toDevice(Map<Integer, NDArray> x, Device device) {
        Map<Integer, NDArray> result = new HashMap<>();
        for (Map.Entry<Integer, NDArray> entry : x.entrySet()) {
            result.put(entry.getKey(), entry.getValue().toDevice(device, false));
        }
        return result;
    }

    /**
     * Add batch dimension to array.
     */
    public static NDArray batchify(NDArray x) {
        return x.expandDims(0);
    }

    /**
     * Apply function to all values in a map.
     */
    public static <K, V, R> Map<K, R> applyDict(Function<V, R> fn, Map<K, V> dict) {
        Map<K, R> result = new HashMap<>();
        for (Map.Entry<K, V> entry : dict.entrySet()) {
            result.put(entry.getKey(), fn.apply(entry.getValue()));
        }
        return result;
    }

    /**
     * Normalize array to [0, 1] range.
     */
    public static NDArray normalize(NDArray x) {
        NDArray min = x.min();
        x = x.sub(min);
        NDArray max = x.max();
        return x.div(max);
    }

    /**
     * Format number with K/M suffix.
     */
    public static String formatNumber(long num) {
        if (num >= 1_000_000) {
            return String.format("%.2f M", num / 1_000_000.0);
        } else {
            return String.format("%.2f k", num / 1_000.0);
        }
    }

    /**
     * Report model parameter counts.
     */
    public static long reportParameters(ai.djl.nn.Block model, int topk) {
        // Get parameter count
        long totalParams = model.getParameters().values().stream()
                .mapToLong(param -> {
                    if (param.getArray() != null) {
                        return param.getArray().size();
                    }
                    return 0;
                })
                .sum();

        System.out.printf("[ utils/arrays ] Total parameters: %s%n", formatNumber(totalParams));

        return totalParams;
    }

    /**
     * Repeat array along specified dimension.
     */
    public static NDArray repeat(NDArray x, int repeats, int axis) {
        return x.repeat(axis, repeats);
    }

    /**
     * Concatenate arrays along specified dimension.
     */
    public static NDArray concat(NDList arrays, int axis) {
        if (arrays.isEmpty()) {
            throw new IllegalArgumentException("Cannot concatenate empty list");
        }
        NDArray result = arrays.get(0);
        for (int i = 1; i < arrays.size(); i++) {
            result = result.concat(arrays.get(i), axis);
        }
        return result;
    }
}
