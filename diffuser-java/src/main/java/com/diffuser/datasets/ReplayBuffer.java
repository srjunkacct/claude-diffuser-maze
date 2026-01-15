package com.diffuser.datasets;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * In-memory replay buffer for storing episode data.
 * Pre-allocates space for efficient storage of fixed-capacity data.
 */
public class ReplayBuffer {

    private final int maxNEpisodes;
    private final int maxPathLength;
    private final float terminationPenalty;

    private final Map<String, float[][][]> data;  // [episode, step, dim]
    private final int[] pathLengths;
    private List<String> keys;
    private int count;

    public ReplayBuffer(int maxNEpisodes, int maxPathLength, float terminationPenalty) {
        this.maxNEpisodes = maxNEpisodes;
        this.maxPathLength = maxPathLength;
        this.terminationPenalty = terminationPenalty;
        this.data = new HashMap<>();
        this.pathLengths = new int[maxNEpisodes];
        this.count = 0;
    }

    /**
     * Add an episode path to the buffer.
     */
    public void addPath(Map<String, float[][]> path) {
        int pathLength = path.get("observations").length;
        if (pathLength > maxPathLength) {
            throw new IllegalArgumentException("Path length exceeds maximum: " + pathLength + " > " + maxPathLength);
        }

        // Set keys on first path
        if (keys == null) {
            keys = new ArrayList<>(path.keySet());
        }

        // Add data for each key
        for (String key : keys) {
            float[][] pathData = path.get(key);
            if (pathData == null) continue;

            int dim = pathData[0].length;

            // Allocate array for this key if needed
            if (!data.containsKey(key)) {
                data.put(key, new float[maxNEpisodes][maxPathLength][dim]);
            }

            // Copy path data
            float[][][] keyData = data.get(key);
            for (int t = 0; t < pathLength; t++) {
                System.arraycopy(pathData[t], 0, keyData[count][t], 0, dim);
            }
        }

        // Handle termination penalty
        if (path.containsKey("terminals")) {
            float[][] terminals = path.get("terminals");
            boolean hasTerminal = false;
            for (float[] t : terminals) {
                if (t[0] > 0.5) {
                    hasTerminal = true;
                    break;
                }
            }
            if (hasTerminal && terminationPenalty != 0) {
                // Check it's not a timeout
                boolean hasTimeout = false;
                if (path.containsKey("timeouts")) {
                    for (float[] t : path.get("timeouts")) {
                        if (t[0] > 0.5) {
                            hasTimeout = true;
                            break;
                        }
                    }
                }
                if (!hasTimeout) {
                    data.get("rewards")[count][pathLength - 1][0] += terminationPenalty;
                }
            }
        }

        pathLengths[count] = pathLength;
        count++;
    }

    /**
     * Finalize the buffer by trimming to actual size.
     */
    public void finalize() {
        // Trim path lengths array
        // In Java we can't resize arrays, but we track the count

        System.out.printf("[ ReplayBuffer ] Finalized replay buffer | %d episodes%n", count);
    }

    /**
     * Get data for a specific key.
     */
    public float[][][] get(String key) {
        return data.get(key);
    }

    /**
     * Get observations.
     */
    public float[][][] getObservations() {
        return data.get("observations");
    }

    /**
     * Get actions.
     */
    public float[][][] getActions() {
        return data.get("actions");
    }

    /**
     * Get rewards.
     */
    public float[][][] getRewards() {
        return data.get("rewards");
    }

    public int[] getPathLengths() {
        int[] result = new int[count];
        System.arraycopy(pathLengths, 0, result, 0, count);
        return result;
    }

    public int getNEpisodes() {
        return count;
    }

    public int getMaxPathLength() {
        return maxPathLength;
    }

    public List<String> getKeys() {
        return keys != null ? new ArrayList<>(keys) : new ArrayList<>();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[ ReplayBuffer ] Fields:\n");
        for (String key : getKeys()) {
            if (!key.equals("path_lengths")) {
                float[][][] keyData = data.get(key);
                if (keyData != null) {
                    sb.append(String.format("    %s: [%d, %d, %d]%n",
                            key, count, maxPathLength, keyData[0][0].length));
                }
            }
        }
        return sb.toString();
    }
}
