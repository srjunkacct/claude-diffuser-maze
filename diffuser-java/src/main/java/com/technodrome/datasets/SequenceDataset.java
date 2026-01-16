package com.technodrome.datasets;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.ndarray.NDList;

import com.technodrome.datasets.normalization.Normalizer;

import java.io.IOException;
import java.util.*;

/**
 * Dataset for trajectory sequences.
 * Loads episodes from files and creates sliding window samples for training.
 */
public class SequenceDataset extends RandomAccessDataset {

    protected final int horizon;
    protected final int maxPathLength;
    protected final boolean usePadding;

    protected final ReplayBuffer fields;
    protected int[][] indices;  // [sample_idx, (episode_idx, start, end)]

    protected Normalizer observationNormalizer;
    protected Normalizer actionNormalizer;

    protected float[][][] normedObservations;
    protected float[][][] normedActions;

    protected final int observationDim;
    protected final int actionDim;
    protected final int nEpisodes;

    protected SequenceDataset(Builder builder) {
        super(builder);

        this.horizon = builder.horizon;
        this.maxPathLength = builder.maxPathLength;
        this.usePadding = builder.usePadding;

        // Load data into replay buffer
        this.fields = new ReplayBuffer(builder.maxNEpisodes, maxPathLength, builder.terminationPenalty);

        for (Map<String, float[][]> episode : builder.episodes) {
            fields.addPath(episode);
        }
        fields.finalize();

        this.nEpisodes = fields.getNEpisodes();
        this.observationDim = fields.getObservations()[0][0].length;
        this.actionDim = fields.getActions()[0][0].length;

        // Create normalizers
        this.observationNormalizer = createNormalizer(builder.normalizerType, flattenData(fields.getObservations()));
        this.actionNormalizer = createNormalizer(builder.normalizerType, flattenData(fields.getActions()));

        // Make indices
        this.indices = makeIndices(fields.getPathLengths(), horizon);

        // Normalize data
        normalize();

        System.out.println(fields);
    }

    private Normalizer createNormalizer(String type, float[][] data) {
        switch (type) {
            case "LimitsNormalizer":
                return new com.technodrome.datasets.normalization.LimitsNormalizer(data);
            case "GaussianNormalizer":
                return new com.technodrome.datasets.normalization.GaussianNormalizer(data);
            case "SafeLimitsNormalizer":
                return new com.technodrome.datasets.normalization.SafeLimitsNormalizer(data);
            default:
                return new com.technodrome.datasets.normalization.LimitsNormalizer(data);
        }
    }

    private float[][] flattenData(float[][][] data) {
        List<float[]> flattened = new ArrayList<>();
        int[] pathLengths = fields.getPathLengths();
        for (int ep = 0; ep < nEpisodes; ep++) {
            for (int t = 0; t < pathLengths[ep]; t++) {
                flattened.add(data[ep][t]);
            }
        }
        return flattened.toArray(new float[0][]);
    }

    protected void normalize() {
        float[][][] observations = fields.getObservations();
        float[][][] actions = fields.getActions();

        normedObservations = new float[nEpisodes][maxPathLength][observationDim];
        normedActions = new float[nEpisodes][maxPathLength][actionDim];

        for (int ep = 0; ep < nEpisodes; ep++) {
            for (int t = 0; t < maxPathLength; t++) {
                normedObservations[ep][t] = observationNormalizer.normalize(observations[ep][t]);
                normedActions[ep][t] = actionNormalizer.normalize(actions[ep][t]);
            }
        }
    }

    protected int[][] makeIndices(int[] pathLengths, int horizon) {
        List<int[]> indexList = new ArrayList<>();

        for (int i = 0; i < pathLengths.length; i++) {
            int pathLength = pathLengths[i];
            int maxStart = Math.min(pathLength - 1, maxPathLength - horizon);
            if (!usePadding) {
                maxStart = Math.min(maxStart, pathLength - horizon);
            }
            for (int start = 0; start < maxStart; start++) {
                int end = start + horizon;
                indexList.add(new int[]{i, start, end});
            }
        }

        return indexList.toArray(new int[0][]);
    }

    /**
     * Get conditions for a trajectory (condition on initial observation).
     */
    protected Map<Integer, float[]> getConditions(float[][] observations) {
        Map<Integer, float[]> conditions = new HashMap<>();
        conditions.put(0, observations[0]);
        return conditions;
    }

    @Override
    public void prepare(ai.djl.util.Progress progress) throws IOException {
        // Data is already loaded in constructor, nothing to prepare
    }

    @Override
    protected long availableSize() {
        return indices.length;
    }

    @Override
    public Record get(NDManager manager, long index) throws IOException {
        int[] idx = indices[(int) index];
        int pathInd = idx[0];
        int start = idx[1];
        int end = idx[2];

        // Get normalized observations and actions for this slice
        float[][] observations = new float[horizon][observationDim];
        float[][] actions = new float[horizon][actionDim];

        for (int t = 0; t < horizon; t++) {
            observations[t] = normedObservations[pathInd][start + t];
            actions[t] = normedActions[pathInd][start + t];
        }

        // Concatenate actions and observations to form trajectory
        float[][] trajectory = new float[horizon][actionDim + observationDim];
        for (int t = 0; t < horizon; t++) {
            System.arraycopy(actions[t], 0, trajectory[t], 0, actionDim);
            System.arraycopy(observations[t], 0, trajectory[t], actionDim, observationDim);
        }

        // Get conditions
        Map<Integer, float[]> conditions = getConditions(observations);

        // Create NDArrays
        NDArray trajArray = manager.create(trajectory);
        NDArray condArray = manager.create(conditions.get(0));

        return new Record(new NDList(trajArray, condArray), new NDList());
    }

    // Getters
    public int getObservationDim() {
        return observationDim;
    }

    public int getActionDim() {
        return actionDim;
    }

    public int getHorizon() {
        return horizon;
    }

    public int getNEpisodes() {
        return nEpisodes;
    }

    public Normalizer getObservationNormalizer() {
        return observationNormalizer;
    }

    public Normalizer getActionNormalizer() {
        return actionNormalizer;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder extends BaseBuilder<Builder> {
        protected int horizon = 64;
        protected int maxPathLength = 1000;
        protected int maxNEpisodes = 10000;
        protected float terminationPenalty = 0;
        protected boolean usePadding = true;
        protected String normalizerType = "LimitsNormalizer";
        protected List<Map<String, float[][]>> episodes = new ArrayList<>();

        public Builder setHorizon(int horizon) {
            this.horizon = horizon;
            return self();
        }

        public Builder setMaxPathLength(int maxPathLength) {
            this.maxPathLength = maxPathLength;
            return self();
        }

        public Builder setMaxNEpisodes(int maxNEpisodes) {
            this.maxNEpisodes = maxNEpisodes;
            return self();
        }

        public Builder setTerminationPenalty(float penalty) {
            this.terminationPenalty = penalty;
            return self();
        }

        public Builder setUsePadding(boolean usePadding) {
            this.usePadding = usePadding;
            return self();
        }

        public Builder setNormalizerType(String normalizerType) {
            this.normalizerType = normalizerType;
            return self();
        }

        public Builder setEpisodes(List<Map<String, float[][]>> episodes) {
            this.episodes = episodes;
            return self();
        }

        @Override
        protected Builder self() {
            return this;
        }

        public SequenceDataset build() {
            return new SequenceDataset(this);
        }
    }
}
