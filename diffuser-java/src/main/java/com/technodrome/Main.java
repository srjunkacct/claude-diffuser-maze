package com.technodrome;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;

import com.technodrome.datasets.DatasetLoader;
import com.technodrome.datasets.SequenceDataset;
import com.technodrome.models.GaussianDiffusion;
import com.technodrome.models.TemporalUnet;
import com.technodrome.training.DiffusionTrainer;

import java.util.List;
import java.util.Map;

/**
 * Main entry point demonstrating diffusion model usage.
 */
public class Main {

    public static void main(String[] args) {
        System.out.println("Diffuser Java - Diffusion Models for Offline RL");
        System.out.println("================================================");

        // Check GPU availability
        Engine engine = Engine.getInstance();
        System.out.printf("Engine: %s%n", engine.getEngineName());
        System.out.printf("GPU available: %b%n", engine.getGpuCount() > 0);
        System.out.printf("GPU count: %d%n", engine.getGpuCount());
        Device device = engine.defaultDevice();
        System.out.printf("Default device: %s%n", device);
        if (!device.isGpu()) {
            System.out.println("WARNING: Running on CPU! Training will be slow.");
        }

        // Configuration
        int horizon = 32;
        int observationDim = 4;
        int actionDim = 2;
        int transitionDim = observationDim + actionDim;

        // Create sample dataset
        System.out.println("\nCreating sample dataset...");
        List<Map<String, float[][]>> episodes = DatasetLoader.createSampleDataset(
                100,  // n_episodes
                64,   // max_path_length
                observationDim,
                actionDim
        );
        System.out.printf("Created %d episodes%n", episodes.size());

        // Build sequence dataset
        System.out.println("\nBuilding sequence dataset...");
        SequenceDataset dataset = SequenceDataset.builder()
                .setEpisodes(episodes)
                .setHorizon(horizon)
                .setMaxPathLength(64)
                .setNormalizerType("LimitsNormalizer")
                .setSampling(32, true)  // batch_size, shuffle
                .build();

        System.out.printf("Dataset size: %d samples%n", dataset.size());
        System.out.printf("Observation dim: %d%n", dataset.getObservationDim());
        System.out.printf("Action dim: %d%n", dataset.getActionDim());

        // Create models
        System.out.println("\nCreating models...");
        try (NDManager manager = NDManager.newBaseManager()) {

            // Create TemporalUnet model
            TemporalUnet unet = new TemporalUnet(
                    horizon,
                    transitionDim,
                    observationDim,  // cond_dim
                    32,  // dim
                    new int[]{1, 2, 4, 8}  // dim_mults
            );

            // Create GaussianDiffusion model
            GaussianDiffusion diffusion = new GaussianDiffusion(
                    unet,
                    horizon,
                    observationDim,
                    actionDim,
                    1000,  // n_timesteps
                    "l1",  // loss_type
                    true,  // clip_denoised
                    true,  // predict_epsilon
                    1.0f,  // action_weight
                    1.0f,  // loss_discount
                    null   // loss_weights
            );

            System.out.println("Models created successfully!");

            // Create trainer
            // Note: Reduce batch size if running out of GPU memory
            System.out.println("\nCreating trainer...");
            int trainingBatchSize = 8;  // Reduced from 32 to help with memory
            DiffusionTrainer trainer = DiffusionTrainer.builder()
                    .setModel(diffusion)
                    .setDataset(dataset)
                    .setBatchSize(trainingBatchSize)
                    .setLearningRate(2e-5f)
                    .setResultsFolder("./results")
                    .build();
            System.out.printf("Training batch size: %d%n", trainingBatchSize);

            System.out.println("Trainer created successfully!");

            // Note: Actual training would be done with:
            trainer.initialize(manager);
            trainer.train(10000);

            System.out.println("\n================================================");
            System.out.println("Setup complete! Ready for training.");
            System.out.println("To train the model, initialize and call train().");

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
