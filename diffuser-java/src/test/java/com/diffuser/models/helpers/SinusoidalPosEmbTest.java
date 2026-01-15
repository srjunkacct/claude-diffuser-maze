package com.diffuser.models.helpers;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SinusoidalPosEmbTest {

    private NDManager manager;
    private ParameterStore parameterStore;

    @BeforeEach
    void setUp() {
        manager = NDManager.newBaseManager();
        parameterStore = new ParameterStore(manager, false);
    }

    @AfterEach
    void tearDown() {
        manager.close();
    }

    @Test
    void testOutputShape() {
        int dim = 32;
        int batchSize = 4;

        SinusoidalPosEmb emb = new SinusoidalPosEmb(dim);
        emb.initialize(manager, ai.djl.ndarray.types.DataType.FLOAT32, new Shape(batchSize));

        NDArray input = manager.arange(batchSize).toType(ai.djl.ndarray.types.DataType.FLOAT32, false);
        NDList output = emb.forward(parameterStore, new NDList(input), false);

        NDArray result = output.singletonOrThrow();
        assertEquals(2, result.getShape().dimension());
        assertEquals(batchSize, result.getShape().get(0));
        assertEquals(dim, result.getShape().get(1));
    }

    @Test
    void testEmbeddingsAreDifferent() {
        int dim = 64;
        SinusoidalPosEmb emb = new SinusoidalPosEmb(dim);
        emb.initialize(manager, ai.djl.ndarray.types.DataType.FLOAT32, new Shape(2));

        NDArray input = manager.create(new float[]{0, 100});
        NDList output = emb.forward(parameterStore, new NDList(input), false);

        NDArray result = output.singletonOrThrow();
        NDArray emb0 = result.get(0);
        NDArray emb100 = result.get(1);

        // Embeddings for different timesteps should be different
        float diff = emb0.sub(emb100).abs().sum().getFloat();
        assertTrue(diff > 0.1f, "Embeddings for different timesteps should differ");
    }
}
