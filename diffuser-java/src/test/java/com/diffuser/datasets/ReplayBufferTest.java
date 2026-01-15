package com.diffuser.datasets;

import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ReplayBufferTest {

    @Test
    void testAddPath() {
        ReplayBuffer buffer = new ReplayBuffer(100, 50, 0);

        // Create a sample episode
        Map<String, float[][]> episode = createSampleEpisode(10, 4, 2);

        buffer.addPath(episode);

        assertEquals(1, buffer.getNEpisodes());
        assertEquals(10, buffer.getPathLengths()[0]);
    }

    @Test
    void testMultiplePaths() {
        ReplayBuffer buffer = new ReplayBuffer(100, 50, 0);

        // Add multiple episodes
        for (int i = 0; i < 5; i++) {
            Map<String, float[][]> episode = createSampleEpisode(10 + i, 4, 2);
            buffer.addPath(episode);
        }

        assertEquals(5, buffer.getNEpisodes());
        assertArrayEquals(new int[]{10, 11, 12, 13, 14}, buffer.getPathLengths());
    }

    @Test
    void testFinalize() {
        ReplayBuffer buffer = new ReplayBuffer(100, 50, 0);

        buffer.addPath(createSampleEpisode(10, 4, 2));
        buffer.addPath(createSampleEpisode(15, 4, 2));
        buffer.finalize();

        assertEquals(2, buffer.getNEpisodes());
    }

    @Test
    void testGetObservations() {
        ReplayBuffer buffer = new ReplayBuffer(100, 50, 0);

        Map<String, float[][]> episode = createSampleEpisode(10, 4, 2);
        buffer.addPath(episode);

        float[][][] observations = buffer.getObservations();
        assertNotNull(observations);
        assertEquals(4, observations[0][0].length);  // observation dim
    }

    @Test
    void testTerminationPenalty() {
        ReplayBuffer buffer = new ReplayBuffer(100, 50, -10.0f);

        Map<String, float[][]> episode = createSampleEpisode(10, 4, 2);
        // Set terminal at last step
        episode.get("terminals")[9][0] = 1.0f;
        episode.get("timeouts")[9][0] = 0.0f;

        buffer.addPath(episode);

        // Check that penalty was applied
        float[][][] rewards = buffer.getRewards();
        assertEquals(-10.0f, rewards[0][9][0], 0.001f);
    }

    private Map<String, float[][]> createSampleEpisode(int length, int obsDim, int actionDim) {
        Map<String, float[][]> episode = new HashMap<>();

        float[][] observations = new float[length][obsDim];
        float[][] actions = new float[length][actionDim];
        float[][] rewards = new float[length][1];
        float[][] terminals = new float[length][1];
        float[][] timeouts = new float[length][1];

        for (int t = 0; t < length; t++) {
            for (int d = 0; d < obsDim; d++) {
                observations[t][d] = (float) Math.random();
            }
            for (int d = 0; d < actionDim; d++) {
                actions[t][d] = (float) Math.random();
            }
            rewards[t][0] = 0;
            terminals[t][0] = 0;
            timeouts[t][0] = 0;
        }

        episode.put("observations", observations);
        episode.put("actions", actions);
        episode.put("rewards", rewards);
        episode.put("terminals", terminals);
        episode.put("timeouts", timeouts);

        return episode;
    }
}
