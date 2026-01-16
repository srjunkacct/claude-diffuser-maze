package com.technodrome.datasets;

import ai.djl.ndarray.NDArray;
import java.util.Map;

/**
 * Batch container for trajectory data.
 */
public class Batch {
    private final NDArray trajectories;
    private final Map<Integer, NDArray> conditions;

    public Batch(NDArray trajectories, Map<Integer, NDArray> conditions) {
        this.trajectories = trajectories;
        this.conditions = conditions;
    }

    public NDArray getTrajectories() {
        return trajectories;
    }

    public Map<Integer, NDArray> getConditions() {
        return conditions;
    }
}

/**
 * Extended batch with value targets for value function training.
 */
class ValueBatch extends Batch {
    private final NDArray values;

    public ValueBatch(NDArray trajectories, Map<Integer, NDArray> conditions, NDArray values) {
        super(trajectories, conditions);
        this.values = values;
    }

    public NDArray getValues() {
        return values;
    }
}
