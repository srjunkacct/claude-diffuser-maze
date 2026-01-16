package com.technodrome.datasets;

import java.util.HashMap;
import java.util.Map;

/**
 * Dataset that conditions on both initial and final observations.
 * Used for goal-conditioned trajectory generation.
 */
public class GoalDataset extends SequenceDataset {

    protected GoalDataset(Builder builder) {
        super(builder);
    }

    @Override
    protected Map<Integer, float[]> getConditions(float[][] observations) {
        Map<Integer, float[]> conditions = new HashMap<>();
        conditions.put(0, observations[0]);
        conditions.put(horizon - 1, observations[observations.length - 1]);
        return conditions;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder extends SequenceDataset.Builder {
        @Override
        protected Builder self() {
            return this;
        }

        @Override
        public GoalDataset build() {
            return new GoalDataset(this);
        }
    }
}
