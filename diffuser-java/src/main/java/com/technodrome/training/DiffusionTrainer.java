package com.technodrome.training;

import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import ai.djl.pytorch.jni.JniUtils;

import com.technodrome.models.GaussianDiffusion;
import com.technodrome.models.helpers.WeightedLoss;
import com.technodrome.utils.Timer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/**
 * Trainer for diffusion models.
 * Handles training loop, EMA updates, checkpointing, and logging.
 */
public class DiffusionTrainer {

    private final GaussianDiffusion model;
    private GaussianDiffusion emaModel;
    private final EMA ema;
    private final RandomAccessDataset dataset;

    private final int batchSize;
    private final int gradientAccumulateEvery;
    private final int stepStartEma;
    private final int updateEmaEvery;
    private final int logFreq;
    private final int saveFreq;

    private final String resultsFolder;
    private final float learningRate;

    private Optimizer optimizer;
    private NDManager manager;
    private ParameterStore parameterStore;

    private int step;

    public DiffusionTrainer(Builder builder) {
        this.model = builder.model;
        this.dataset = builder.dataset;
        this.batchSize = builder.batchSize;
        this.gradientAccumulateEvery = builder.gradientAccumulateEvery;
        this.stepStartEma = builder.stepStartEma;
        this.updateEmaEvery = builder.updateEmaEvery;
        this.logFreq = builder.logFreq;
        this.saveFreq = builder.saveFreq;
        this.resultsFolder = builder.resultsFolder;
        this.learningRate = builder.learningRate;

        this.ema = new EMA(builder.emaDecay);
        this.step = 0;
    }

    /**
     * Initialize the trainer.
     */
    public void initialize(NDManager manager) throws IOException {
        this.manager = manager;
        this.parameterStore = new ParameterStore(manager, false);

        // Initialize model parameters
        Shape inputShape = new Shape(batchSize, model.getHorizon(), model.getTransitionDim());
        model.initialize(manager, DataType.FLOAT32, inputShape);
        model.initializeParameters(manager);

        // Enable gradient tracking on trainable parameters
        // Use blacklist approach: enable gradients on everything EXCEPT known non-trainable params
        int enabledCount = 0;
        int skippedCount = 0;
        System.out.println("[ training ] Processing parameters:");
        for (var pair : model.getParameters()) {
            String paramName = pair.getKey();
            String paramNameLower = paramName.toLowerCase();
            // Skip BatchNorm running statistics and tracking counters
            // These cause "not differentiable with respect to running_mean" errors
            boolean isNonTrainable = paramNameLower.contains("running_mean") ||
                                     paramNameLower.contains("running_var") ||
                                     paramNameLower.contains("runningmean") ||
                                     paramNameLower.contains("runningvar") ||
                                     paramNameLower.contains("num_batches") ||
                                     paramNameLower.contains("numbatches");
            if (!isNonTrainable) {
                pair.getValue().getArray().setRequiresGradient(true);
                enabledCount++;
                if (enabledCount <= 10) {
                    System.out.printf("  [enabled] %s%n", paramName);
                }
            } else {
                skippedCount++;
                System.out.printf("  [skipped] %s%n", paramName);
            }
        }
        System.out.printf("[ training ] Enabled gradients on %d parameters, skipped %d non-trainable%n",
                          enabledCount, skippedCount);

        // Create EMA model as a deep copy
        // Note: In DJL, creating a true deep copy of a Block is complex
        // For simplicity, we'll use the same model structure
        // In production, you would serialize/deserialize to create a true copy

        // Create optimizer
        this.optimizer = Optimizer.adam()
                .optLearningRateTracker(Tracker.fixed(learningRate))
                .build();

        // Create results folder
        Path resultsPath = Paths.get(resultsFolder);
        if (!Files.exists(resultsPath)) {
            Files.createDirectories(resultsPath);
        }

        resetParameters();
    }

    /**
     * Reset EMA model to current model parameters.
     */
    private void resetParameters() {
        // In a full implementation, copy parameters from model to emaModel
    }

    /**
     * Update EMA model parameters.
     */
    private void stepEma() {
        if (step < stepStartEma) {
            resetParameters();
            return;
        }
        if (emaModel != null) {
            ema.updateModelAverage(emaModel, model);
        }
    }

    /**
     * Run training loop.
     * Uses global manager - tensors are freed by PyTorch's native memory management
     * when no longer referenced, plus explicit CUDA cache clearing.
     */
    public void train(int nTrainSteps) throws IOException, TranslateException {
        Timer timer = new Timer();

        for (int trainStep = 0; trainStep < nTrainSteps; trainStep++) {
            float lossValue;

            // Get batch data using global manager
            Iterator<Batch> dataIterator = dataset.getData(manager).iterator();
            if (!dataIterator.hasNext()) {
                continue; // Skip if no data
            }
            Batch batch = dataIterator.next();

            // Single forward-backward pass
            try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                NDList data = batch.getData();
                NDArray trajectories = data.get(0);
                NDArray conditions = data.get(1);

                // Create conditions map
                Map<Integer, NDArray> condMap = new HashMap<>();
                condMap.put(0, conditions);

                // Compute loss
                WeightedLoss.LossResult result = model.loss(parameterStore, trajectories, condMap, true);
                lossValue = result.loss.getFloat();

                // Debug: print loss info on first step
                if (step == 0) {
                    System.out.printf("[ debug ] Loss shape: %s, hasGradient: %b%n",
                                      result.loss.getShape(), result.loss.hasGradient());
                }

                // Backward pass
                gc.backward(result.loss);
            } // GradientCollector closes here - releases computation graph

            // Close batch to free batch tensors
            batch.close();

            // Update parameters using simple SGD (all in-place operations)
            for (var pair : model.getParameters()) {
                NDArray param = pair.getValue().getArray();
                if (param.hasGradient()) {
                    NDArray grad = param.getGradient();
                    // In-place: grad *= lr, then param -= grad
                    grad.muli(learningRate);
                    param.subi(grad);
                    // Zero gradient (in-place) for next iteration
                    grad.muli(0);
                }
            }

            // Clear CUDA cache to return freed memory to allocator
            JniUtils.emptyCudaCache();

            // Update EMA
            if (step % updateEmaEvery == 0) {
                stepEma();
            }

            // Save checkpoint
            if (step % saveFreq == 0) {
                save(step);
            }

            // Log progress
            if (step % logFreq == 0) {
                double elapsed = timer.elapsed();

                // Memory stats
                Runtime runtime = Runtime.getRuntime();
                long usedMemory = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024);
                long maxMemory = runtime.maxMemory() / (1024 * 1024);

                System.out.printf("%d: %.4f | t: %.4f | mem: %dMB/%dMB%n",
                                  step, lossValue, elapsed, usedMemory, maxMemory);

                // Periodic garbage collection
                System.gc();
            }

            step++;
        }
    }

    private Iterator<Batch> createDataIterator() throws IOException, TranslateException {
        return dataset.getData(manager).iterator();
    }

    /**
     * Save model checkpoint.
     */
    public void save(int epoch) throws IOException {
        Path savePath = Paths.get(resultsFolder, String.format("state_%d.pt", epoch));

        try (Model djlModel = Model.newInstance("diffusion")) {
            djlModel.setBlock(model);
            djlModel.save(savePath.getParent(), "state_" + epoch);
        }

        System.out.printf("[ training ] Saved model to %s%n", savePath);
    }

    /**
     * Load model checkpoint.
     */
    public void load(int epoch) throws IOException, ai.djl.MalformedModelException {
        Path loadPath = Paths.get(resultsFolder, String.format("state_%d", epoch));

        try (Model djlModel = Model.newInstance("diffusion")) {
            djlModel.setBlock(model);
            djlModel.load(loadPath.getParent(), "state_" + epoch);
        }

        System.out.printf("[ training ] Loaded model from %s%n", loadPath);
    }

    // Getters
    public int getStep() {
        return step;
    }

    public GaussianDiffusion getModel() {
        return model;
    }

    public GaussianDiffusion getEmaModel() {
        return emaModel != null ? emaModel : model;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private GaussianDiffusion model;
        private RandomAccessDataset dataset;
        private float emaDecay = 0.995f;
        private int batchSize = 32;
        private float learningRate = 2e-5f;
        private int gradientAccumulateEvery = 2;
        private int stepStartEma = 2000;
        private int updateEmaEvery = 10;
        private int logFreq = 100;
        private int saveFreq = 1000;
        private String resultsFolder = "./results";

        public Builder setModel(GaussianDiffusion model) {
            this.model = model;
            return this;
        }

        public Builder setDataset(RandomAccessDataset dataset) {
            this.dataset = dataset;
            return this;
        }

        public Builder setEmaDecay(float emaDecay) {
            this.emaDecay = emaDecay;
            return this;
        }

        public Builder setBatchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public Builder setLearningRate(float learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder setGradientAccumulateEvery(int gradientAccumulateEvery) {
            this.gradientAccumulateEvery = gradientAccumulateEvery;
            return this;
        }

        public Builder setStepStartEma(int stepStartEma) {
            this.stepStartEma = stepStartEma;
            return this;
        }

        public Builder setUpdateEmaEvery(int updateEmaEvery) {
            this.updateEmaEvery = updateEmaEvery;
            return this;
        }

        public Builder setLogFreq(int logFreq) {
            this.logFreq = logFreq;
            return this;
        }

        public Builder setSaveFreq(int saveFreq) {
            this.saveFreq = saveFreq;
            return this;
        }

        public Builder setResultsFolder(String resultsFolder) {
            this.resultsFolder = resultsFolder;
            return this;
        }

        public DiffusionTrainer build() {
            if (model == null) {
                throw new IllegalStateException("Model must be set");
            }
            if (dataset == null) {
                throw new IllegalStateException("Dataset must be set");
            }
            return new DiffusionTrainer(this);
        }
    }
}
