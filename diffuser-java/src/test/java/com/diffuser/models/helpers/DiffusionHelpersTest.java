package com.diffuser.models.helpers;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class DiffusionHelpersTest {

    private NDManager manager;

    @BeforeEach
    void setUp() {
        manager = NDManager.newBaseManager();
    }

    @AfterEach
    void tearDown() {
        manager.close();
    }

    @Test
    void testCosineBetaSchedule() {
        int timesteps = 1000;
        NDArray betas = DiffusionHelpers.cosineBetaSchedule(manager, timesteps);

        // Check shape
        assertEquals(new Shape(timesteps), betas.getShape());

        // Betas should be between 0 and 1
        float minBeta = betas.min().getFloat();
        float maxBeta = betas.max().getFloat();
        assertTrue(minBeta >= 0, "Min beta should be >= 0");
        assertTrue(maxBeta <= 1, "Max beta should be <= 1");

        // Betas should generally increase (cosine schedule)
        float firstBeta = betas.getFloat(0);
        float lastBeta = betas.getFloat(timesteps - 1);
        assertTrue(lastBeta > firstBeta, "Betas should increase over time");
    }

    @Test
    void testExtract() {
        // Create a simple array of values
        NDArray a = manager.create(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f});

        // Create timestep indices
        NDArray t = manager.create(new long[]{0, 2, 4});

        // Target shape for broadcasting
        Shape xShape = new Shape(3, 4, 5);

        NDArray result = DiffusionHelpers.extract(a, t, xShape);

        // Check shape is correct for broadcasting
        assertEquals(new Shape(3, 1, 1), result.getShape());

        // Check values are correctly extracted
        float[] expected = {0.1f, 0.3f, 0.5f};
        float[] actual = result.flatten().toFloatArray();
        assertArrayEquals(expected, actual, 0.001f);
    }

    @Test
    void testApplyConditioning() {
        // Create trajectory [batch=2, horizon=4, transition_dim=5 (action=2, obs=3)]
        NDArray x = manager.zeros(new Shape(2, 4, 5));

        // Create conditions: set observation at t=0
        Map<Integer, NDArray> conditions = new HashMap<>();
        NDArray obsCondition = manager.ones(new Shape(2, 3));  // [batch, obs_dim]
        conditions.put(0, obsCondition);

        int actionDim = 2;
        NDArray result = DiffusionHelpers.applyConditioning(x, conditions, actionDim);

        // Check that conditioning was applied at t=0, starting at actionDim
        NDArray conditioned = result.get(":, 0, 2:");
        float sum = conditioned.sum().getFloat();
        assertEquals(6.0f, sum, 0.001f);  // 2 batches * 3 obs_dim * 1.0

        // Check that other positions are still zero
        NDArray actions = result.get(":, 0, :2");
        assertEquals(0.0f, actions.sum().getFloat(), 0.001f);
    }
}
