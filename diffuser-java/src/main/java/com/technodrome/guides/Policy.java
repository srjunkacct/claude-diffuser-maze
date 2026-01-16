package com.technodrome.guides;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.ParameterStore;

import com.technodrome.datasets.normalization.Normalizer;
import com.technodrome.models.GaussianDiffusion;

import java.util.HashMap;
import java.util.Map;

/**
 * Policy wrapper for diffusion model inference.
 * Handles normalization/denormalization and action extraction.
 */
public class Policy {

    private final GaussianDiffusion diffusionModel;
    private final Normalizer observationNormalizer;
    private final Normalizer actionNormalizer;
    private final int actionDim;
    private final ParameterStore parameterStore;
    private final NDManager manager;

    public Policy(GaussianDiffusion diffusionModel, Normalizer observationNormalizer,
                  Normalizer actionNormalizer, NDManager manager) {
        this.diffusionModel = diffusionModel;
        this.observationNormalizer = observationNormalizer;
        this.actionNormalizer = actionNormalizer;
        this.actionDim = diffusionModel.getActionDim();
        this.manager = manager;
        this.parameterStore = new ParameterStore(manager, false);
    }

    /**
     * Get device of the diffusion model.
     */
    public Device getDevice() {
        return manager.getDevice();
    }

    /**
     * Format and normalize conditions for the model.
     */
    private Map<Integer, NDArray> formatConditions(Map<Integer, float[]> conditions, int batchSize) {
        Map<Integer, NDArray> result = new HashMap<>();

        for (Map.Entry<Integer, float[]> entry : conditions.entrySet()) {
            int timestep = entry.getKey();
            float[] obs = entry.getValue();

            // Normalize observation
            float[] normedObs = observationNormalizer.normalize(obs);

            // Create NDArray and repeat for batch
            NDArray obsArray = manager.create(normedObs);

            // Expand to batch size
            if (batchSize > 1) {
                obsArray = obsArray.expandDims(0).repeat(0, batchSize).squeeze();
            }

            result.put(timestep, obsArray);
        }

        return result;
    }

    /**
     * Generate action from observation.
     *
     * @param conditions map of timestep to observation
     * @return PolicyResult containing action and full trajectories
     */
    public PolicyResult call(Map<Integer, float[]> conditions) {
        return call(conditions, 1);
    }

    /**
     * Generate action from observation with specified batch size.
     *
     * @param conditions map of timestep to observation
     * @param batchSize  number of samples to generate
     * @return PolicyResult containing action and full trajectories
     */
    public PolicyResult call(Map<Integer, float[]> conditions, int batchSize) {
        // Format and normalize conditions
        Map<Integer, NDArray> formattedConditions = formatConditions(conditions, batchSize);

        // Run reverse diffusion process
        NDArray sample = diffusionModel.conditionalSample(parameterStore, formattedConditions, false);

        // Convert to float array for processing
        float[][][] sampleArray = toFloatArray3D(sample);

        // Extract actions [batch, horizon, actionDim]
        float[][][] actions = new float[batchSize][diffusionModel.getHorizon()][actionDim];
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < diffusionModel.getHorizon(); t++) {
                for (int a = 0; a < actionDim; a++) {
                    actions[b][t][a] = sampleArray[b][t][a];
                }
            }
        }

        // Unnormalize actions
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < diffusionModel.getHorizon(); t++) {
                actions[b][t] = actionNormalizer.unnormalize(actions[b][t]);
            }
        }

        // Extract first action from first sample
        float[] action = actions[0][0];

        // Extract observations [batch, horizon, obsDim]
        int obsDim = diffusionModel.getObservationDim();
        float[][][] observations = new float[batchSize][diffusionModel.getHorizon()][obsDim];
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < diffusionModel.getHorizon(); t++) {
                for (int o = 0; o < obsDim; o++) {
                    observations[b][t][o] = sampleArray[b][t][actionDim + o];
                }
            }
        }

        // Unnormalize observations
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < diffusionModel.getHorizon(); t++) {
                observations[b][t] = observationNormalizer.unnormalize(observations[b][t]);
            }
        }

        return new PolicyResult(action, new Trajectories(actions, observations));
    }

    private float[][][] toFloatArray3D(NDArray x) {
        long[] shape = x.getShape().getShape();
        int d0 = (int) shape[0];
        int d1 = (int) shape[1];
        int d2 = (int) shape[2];
        float[] flat = x.toFloatArray();

        float[][][] result = new float[d0][d1][d2];
        int idx = 0;
        for (int i = 0; i < d0; i++) {
            for (int j = 0; j < d1; j++) {
                for (int k = 0; k < d2; k++) {
                    result[i][j][k] = flat[idx++];
                }
            }
        }
        return result;
    }

    /**
     * Container for generated trajectories.
     */
    public static class Trajectories {
        public final float[][][] actions;
        public final float[][][] observations;

        public Trajectories(float[][][] actions, float[][][] observations) {
            this.actions = actions;
            this.observations = observations;
        }
    }

    /**
     * Result container for policy inference.
     */
    public static class PolicyResult {
        public final float[] action;
        public final Trajectories trajectories;

        public PolicyResult(float[] action, Trajectories trajectories) {
            this.action = action;
            this.trajectories = trajectories;
        }
    }
}
