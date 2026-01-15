package com.diffuser.models;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import com.diffuser.models.helpers.DiffusionHelpers;
import com.diffuser.models.helpers.WeightedLoss;
import com.diffuser.utils.Progress;

import java.util.HashMap;
import java.util.Map;

/**
 * Gaussian Diffusion Model for trajectory generation.
 * Implements the forward (noising) and reverse (denoising) diffusion processes.
 */
public class GaussianDiffusion extends AbstractBlock {

    private static final byte VERSION = 1;

    private final AbstractBlock model;
    private final int horizon;
    private final int observationDim;
    private final int actionDim;
    private final int transitionDim;
    private final int nTimesteps;
    private final boolean clipDenoised;
    private final boolean predictEpsilon;
    private final float actionWeight;

    private WeightedLoss lossFn;
    private final String lossType;
    private final float lossDiscount;
    private final Map<Integer, Float> lossWeightsDict;

    // Diffusion parameters (registered as buffers)
    private NDArray betas;
    private NDArray alphasCumprod;
    private NDArray alphasCumprodPrev;
    private NDArray sqrtAlphasCumprod;
    private NDArray sqrtOneMinusAlphasCumprod;
    private NDArray logOneMinusAlphasCumprod;
    private NDArray sqrtRecipAlphasCumprod;
    private NDArray sqrtRecipm1AlphasCumprod;
    private NDArray posteriorVariance;
    private NDArray posteriorLogVarianceClipped;
    private NDArray posteriorMeanCoef1;
    private NDArray posteriorMeanCoef2;
    private NDArray lossWeights;

    public GaussianDiffusion(AbstractBlock model, int horizon, int observationDim, int actionDim) {
        this(model, horizon, observationDim, actionDim, 1000, "l1", false, true, 1.0f, 1.0f, null);
    }

    public GaussianDiffusion(AbstractBlock model, int horizon, int observationDim, int actionDim,
                             int nTimesteps, String lossType, boolean clipDenoised,
                             boolean predictEpsilon, float actionWeight, float lossDiscount,
                             Map<Integer, Float> lossWeightsDict) {
        super(VERSION);

        this.model = model;
        addChildBlock("model", model);

        this.horizon = horizon;
        this.observationDim = observationDim;
        this.actionDim = actionDim;
        this.transitionDim = observationDim + actionDim;
        this.nTimesteps = nTimesteps;
        this.clipDenoised = clipDenoised;
        this.predictEpsilon = predictEpsilon;
        this.actionWeight = actionWeight;
        this.lossType = lossType;
        this.lossDiscount = lossDiscount;
        this.lossWeightsDict = lossWeightsDict != null ? lossWeightsDict : new HashMap<>();
    }

    /**
     * Initialize diffusion parameters. Must be called before training/inference.
     */
    public void initializeParameters(NDManager manager) {
        // Compute beta schedule
        this.betas = DiffusionHelpers.cosineBetaSchedule(manager, nTimesteps);

        NDArray alphas = betas.neg().add(1);
        this.alphasCumprod = alphas.cumSum(0);
        // Fix: cumulative product, not sum
        this.alphasCumprod = cumprod(alphas);

        NDArray ones = manager.ones(new Shape(1)).toType(DataType.FLOAT32, false);
        this.alphasCumprodPrev = ones.concat(alphasCumprod.get(":-1"));

        // Calculations for diffusion q(x_t | x_{t-1}) and others
        this.sqrtAlphasCumprod = alphasCumprod.sqrt();
        this.sqrtOneMinusAlphasCumprod = alphasCumprod.neg().add(1).sqrt();
        this.logOneMinusAlphasCumprod = alphasCumprod.neg().add(1).log();
        this.sqrtRecipAlphasCumprod = alphasCumprod.pow(-0.5);
        this.sqrtRecipm1AlphasCumprod = alphasCumprod.pow(-1).sub(1).sqrt();

        // Calculations for posterior q(x_{t-1} | x_t, x_0)
        NDArray numerator = betas.mul(alphasCumprodPrev.neg().add(1));
        NDArray denominator = alphasCumprod.neg().add(1);
        this.posteriorVariance = numerator.div(denominator);

        // Log calculation clipped because posterior variance is 0 at start
        this.posteriorLogVarianceClipped = posteriorVariance.clip(1e-20, Double.MAX_VALUE).log();

        this.posteriorMeanCoef1 = betas.mul(alphasCumprodPrev.sqrt()).div(alphasCumprod.neg().add(1));
        NDArray alphasSqrt = alphas.sqrt();
        this.posteriorMeanCoef2 = alphasCumprodPrev.neg().add(1).mul(alphasSqrt).div(alphasCumprod.neg().add(1));

        // Get loss coefficients and initialize objective
        this.lossWeights = getLossWeights(manager, actionWeight, lossDiscount, lossWeightsDict);

        // Initialize loss function
        if (lossType.equals("l1")) {
            this.lossFn = new WeightedLoss.WeightedL1(lossWeights, actionDim);
        } else if (lossType.equals("l2")) {
            this.lossFn = new WeightedLoss.WeightedL2(lossWeights, actionDim);
        } else {
            throw new IllegalArgumentException("Unknown loss type: " + lossType);
        }
    }

    private NDArray cumprod(NDArray arr) {
        NDManager manager = arr.getManager();
        float[] values = arr.toFloatArray();
        float[] result = new float[values.length];
        result[0] = values[0];
        for (int i = 1; i < values.length; i++) {
            result[i] = result[i - 1] * values[i];
        }
        return manager.create(result);
    }

    private NDArray getLossWeights(NDManager manager, float actionWeight, float discount,
                                   Map<Integer, Float> weightsDict) {
        // Set loss coefficients for dimensions
        float[] dimWeights = new float[transitionDim];
        for (int i = 0; i < transitionDim; i++) {
            dimWeights[i] = 1.0f;
        }

        // Apply custom weights for observation dimensions
        for (Map.Entry<Integer, Float> entry : weightsDict.entrySet()) {
            int ind = entry.getKey();
            float w = entry.getValue();
            dimWeights[actionDim + ind] *= w;
        }

        // Decay loss with trajectory timestep: discount**t
        float[] discounts = new float[horizon];
        for (int t = 0; t < horizon; t++) {
            discounts[t] = (float) Math.pow(discount, t);
        }

        // Normalize discounts
        float discountMean = 0;
        for (float d : discounts) {
            discountMean += d;
        }
        discountMean /= horizon;
        for (int t = 0; t < horizon; t++) {
            discounts[t] /= discountMean;
        }

        // Outer product: loss_weights = einsum('h,t->ht', discounts, dim_weights)
        float[][] weights = new float[horizon][transitionDim];
        for (int h = 0; h < horizon; h++) {
            for (int t = 0; t < transitionDim; t++) {
                weights[h][t] = discounts[h] * dimWeights[t];
            }
        }

        // Manually set a0 weight
        for (int a = 0; a < actionDim; a++) {
            weights[0][a] = actionWeight;
        }

        return manager.create(weights);
    }

    // ------------------------------------------ sampling ------------------------------------------

    /**
     * Predict x_start from noise prediction.
     */
    private NDArray predictStartFromNoise(NDArray xT, NDArray t, NDArray noise) {
        if (predictEpsilon) {
            NDArray coef1 = DiffusionHelpers.extract(sqrtRecipAlphasCumprod, t, xT.getShape());
            NDArray coef2 = DiffusionHelpers.extract(sqrtRecipm1AlphasCumprod, t, xT.getShape());
            return coef1.mul(xT).sub(coef2.mul(noise));
        } else {
            return noise;
        }
    }

    /**
     * Compute posterior q(x_{t-1} | x_t, x_0).
     */
    private NDArray[] qPosterior(NDArray xStart, NDArray xT, NDArray t) {
        NDArray coef1 = DiffusionHelpers.extract(posteriorMeanCoef1, t, xT.getShape());
        NDArray coef2 = DiffusionHelpers.extract(posteriorMeanCoef2, t, xT.getShape());
        NDArray posteriorMean = coef1.mul(xStart).add(coef2.mul(xT));

        NDArray postVar = DiffusionHelpers.extract(posteriorVariance, t, xT.getShape());
        NDArray postLogVarClipped = DiffusionHelpers.extract(posteriorLogVarianceClipped, t, xT.getShape());

        return new NDArray[]{posteriorMean, postVar, postLogVarClipped};
    }

    /**
     * Compute p(x_{t-1} | x_t) mean and variance.
     */
    private NDArray[] pMeanVariance(ParameterStore parameterStore, NDArray x,
                                    Map<Integer, NDArray> cond, NDArray t, boolean training) {
        // Get model prediction
        NDList modelInputs = new NDList(x, createCondTensor(x.getManager(), cond), t);
        NDList modelOut = model.forward(parameterStore, modelInputs, training);
        NDArray noise = modelOut.singletonOrThrow();

        NDArray xRecon = predictStartFromNoise(x, t, noise);

        if (clipDenoised) {
            xRecon = xRecon.clip(-1, 1);
        }

        NDArray[] posterior = qPosterior(xRecon, x, t);
        return posterior;
    }

    /**
     * Sample x_{t-1} from p(x_{t-1} | x_t).
     */
    private NDArray pSample(ParameterStore parameterStore, NDArray x,
                            Map<Integer, NDArray> cond, NDArray t, boolean training) {
        NDManager manager = x.getManager();
        long batchSize = x.getShape().get(0);

        NDArray[] meanVar = pMeanVariance(parameterStore, x, cond, t, training);
        NDArray modelMean = meanVar[0];
        NDArray modelLogVariance = meanVar[2];

        NDArray noise = manager.randomNormal(x.getShape());

        // No noise when t == 0
        NDArray tZero = t.eq(0).toType(DataType.FLOAT32, false);
        NDArray nonzeroMask = tZero.neg().add(1);

        // Reshape mask for broadcasting
        long[] maskShape = new long[x.getShape().dimension()];
        maskShape[0] = batchSize;
        for (int i = 1; i < maskShape.length; i++) {
            maskShape[i] = 1;
        }
        nonzeroMask = nonzeroMask.reshape(maskShape);

        return modelMean.add(nonzeroMask.mul(modelLogVariance.mul(0.5).exp()).mul(noise));
    }

    /**
     * Full reverse diffusion sampling loop.
     */
    public NDArray pSampleLoop(ParameterStore parameterStore, Shape shape,
                               Map<Integer, NDArray> cond, boolean verbose, boolean training) {
        NDManager manager = betas.getManager();
        long batchSize = shape.get(0);

        // Start from pure noise
        NDArray x = manager.randomNormal(shape);
        x = DiffusionHelpers.applyConditioning(x, cond, actionDim);

        Progress progress = verbose ? new Progress(nTimesteps) : null;

        for (int i = nTimesteps - 1; i >= 0; i--) {
            NDArray timesteps = manager.full(new Shape(batchSize), i).toType(DataType.INT64, false);
            x = pSample(parameterStore, x, cond, timesteps, training);
            x = DiffusionHelpers.applyConditioning(x, cond, actionDim);

            if (progress != null) {
                Map<String, Object> params = new HashMap<>();
                params.put("t", i);
                progress.update(params);
            }
        }

        if (progress != null) {
            progress.close();
        }

        return x;
    }

    /**
     * Generate trajectories conditioned on observations.
     */
    public NDArray conditionalSample(ParameterStore parameterStore, Map<Integer, NDArray> cond,
                                     boolean training) {
        return conditionalSample(parameterStore, cond, horizon, true, training);
    }

    public NDArray conditionalSample(ParameterStore parameterStore, Map<Integer, NDArray> cond,
                                     int horizonOverride, boolean verbose, boolean training) {
        NDManager manager = betas.getManager();
        long batchSize = cond.get(0).getShape().get(0);
        Shape shape = new Shape(batchSize, horizonOverride, transitionDim);
        return pSampleLoop(parameterStore, shape, cond, verbose, training);
    }

    // ------------------------------------------ training ------------------------------------------

    /**
     * Forward diffusion process: add noise to x_start.
     */
    public NDArray qSample(NDArray xStart, NDArray t, NDArray noise) {
        if (noise == null) {
            noise = xStart.getManager().randomNormal(xStart.getShape());
        }

        NDArray coef1 = DiffusionHelpers.extract(sqrtAlphasCumprod, t, xStart.getShape());
        NDArray coef2 = DiffusionHelpers.extract(sqrtOneMinusAlphasCumprod, t, xStart.getShape());

        return coef1.mul(xStart).add(coef2.mul(noise));
    }

    /**
     * Compute training loss.
     */
    public WeightedLoss.LossResult pLosses(ParameterStore parameterStore, NDArray xStart,
                                           Map<Integer, NDArray> cond, NDArray t, boolean training) {
        NDManager manager = xStart.getManager();
        NDArray noise = manager.randomNormal(xStart.getShape());

        NDArray xNoisy = qSample(xStart, t, noise);
        xNoisy = DiffusionHelpers.applyConditioning(xNoisy, cond, actionDim);

        // Get model prediction
        NDList modelInputs = new NDList(xNoisy, createCondTensor(manager, cond), t);
        NDList modelOut = model.forward(parameterStore, modelInputs, training);
        NDArray xRecon = modelOut.singletonOrThrow();
        xRecon = DiffusionHelpers.applyConditioning(xRecon, cond, actionDim);

        if (predictEpsilon) {
            return lossFn.forward(xRecon, noise);
        } else {
            return lossFn.forward(xRecon, xStart);
        }
    }

    /**
     * Compute loss for a batch.
     */
    public WeightedLoss.LossResult loss(ParameterStore parameterStore, NDArray x,
                                        Map<Integer, NDArray> cond, boolean training) {
        NDManager manager = x.getManager();
        long batchSize = x.getShape().get(0);
        NDArray t = manager.randomUniform(0, nTimesteps, new Shape(batchSize))
                .toType(DataType.INT64, false);
        return pLosses(parameterStore, x, cond, t, training);
    }

    private NDArray createCondTensor(NDManager manager, Map<Integer, NDArray> cond) {
        // Create a simple condition tensor for the model
        // This is a placeholder - the actual conditioning depends on model architecture
        if (cond.isEmpty()) {
            return manager.zeros(new Shape(1));
        }
        return cond.values().iterator().next();
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs,
                                     boolean training, PairList<String, Object> params) {
        // Forward pass is conditional sampling
        Map<Integer, NDArray> cond = new HashMap<>();
        cond.put(0, inputs.singletonOrThrow());
        return new NDList(conditionalSample(parameterStore, cond, training));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        long batchSize = inputShapes[0].get(0);
        return new Shape[]{new Shape(batchSize, horizon, transitionDim)};
    }

    // Getters
    public int getHorizon() {
        return horizon;
    }

    public int getObservationDim() {
        return observationDim;
    }

    public int getActionDim() {
        return actionDim;
    }

    public int getTransitionDim() {
        return transitionDim;
    }

    public int getNTimesteps() {
        return nTimesteps;
    }

    public AbstractBlock getModel() {
        return model;
    }
}
