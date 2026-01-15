package com.diffuser.datasets;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.Record;

import java.io.IOException;
import java.util.Map;

/**
 * Dataset that includes value targets for training value functions.
 * Computes discounted cumulative rewards for each trajectory.
 */
public class ValueDataset extends SequenceDataset {

    private final float discount;
    private final float[] discounts;

    protected ValueDataset(ValueBuilder builder) {
        super(builder);
        this.discount = builder.discount;

        // Pre-compute discount factors
        this.discounts = new float[maxPathLength];
        for (int t = 0; t < maxPathLength; t++) {
            discounts[t] = (float) Math.pow(discount, t);
        }
    }

    @Override
    public Record get(NDManager manager, long index) throws IOException {
        // Get base record
        Record baseRecord = super.get(manager, index);

        int[] idx = indices[(int) index];
        int pathInd = idx[0];
        int start = idx[1];

        // Compute discounted return from start position
        float[][][] rewards = fields.getRewards();
        int[] pathLengths = fields.getPathLengths();
        int pathLength = pathLengths[pathInd];

        float value = 0;
        for (int t = start; t < pathLength; t++) {
            int discountIdx = t - start;
            if (discountIdx < discounts.length) {
                value += discounts[discountIdx] * rewards[pathInd][t][0];
            }
        }

        NDArray valueArray = manager.create(new float[]{value});

        // Return record with value included
        NDList data = new NDList(baseRecord.getData());
        data.add(valueArray);

        return new Record(data, new NDList());
    }

    public float getDiscount() {
        return discount;
    }

    public static ValueBuilder builder() {
        return new ValueBuilder();
    }

    public static class ValueBuilder extends SequenceDataset.Builder {
        protected float discount = 0.99f;

        public ValueBuilder setDiscount(float discount) {
            this.discount = discount;
            return this;
        }

        @Override
        protected ValueBuilder self() {
            return this;
        }

        @Override
        public ValueDataset build() {
            return new ValueDataset(this);
        }
    }
}
