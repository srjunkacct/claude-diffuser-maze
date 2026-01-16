package com.technodrome.datasets;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class DatasetLoaderTest {

    @Test
    void testCreateSampleDataset() {
        int nEpisodes = 10;
        int maxPathLength = 50;
        int observationDim = 4;
        int actionDim = 2;

        List<Map<String, float[][]>> episodes = DatasetLoader.createSampleDataset(
                nEpisodes, maxPathLength, observationDim, actionDim);

        assertEquals(nEpisodes, episodes.size());

        // Check first episode
        Map<String, float[][]> episode = episodes.get(0);
        assertTrue(episode.containsKey("observations"));
        assertTrue(episode.containsKey("actions"));
        assertTrue(episode.containsKey("rewards"));
        assertTrue(episode.containsKey("terminals"));
        assertTrue(episode.containsKey("timeouts"));

        // Check dimensions
        float[][] observations = episode.get("observations");
        assertEquals(observationDim, observations[0].length);

        float[][] actions = episode.get("actions");
        assertEquals(actionDim, actions[0].length);
    }

    @Test
    void testSampleDatasetPathLengths() {
        int nEpisodes = 20;
        int maxPathLength = 100;

        List<Map<String, float[][]>> episodes = DatasetLoader.createSampleDataset(
                nEpisodes, maxPathLength, 4, 2);

        for (Map<String, float[][]> episode : episodes) {
            int pathLength = episode.get("observations").length;
            // Path length should be between maxPathLength/2 and maxPathLength
            assertTrue(pathLength >= maxPathLength / 2);
            assertTrue(pathLength <= maxPathLength);
        }
    }

    @Test
    void testSampleDatasetValueRanges() {
        List<Map<String, float[][]>> episodes = DatasetLoader.createSampleDataset(
                5, 50, 4, 2);

        for (Map<String, float[][]> episode : episodes) {
            // Check observations are in [-1, 1]
            for (float[] obs : episode.get("observations")) {
                for (float val : obs) {
                    assertTrue(val >= -1 && val <= 1);
                }
            }

            // Check actions are in [-1, 1]
            for (float[] action : episode.get("actions")) {
                for (float val : action) {
                    assertTrue(val >= -1 && val <= 1);
                }
            }

            // Check rewards are in [0, 1]
            for (float[] reward : episode.get("rewards")) {
                assertTrue(reward[0] >= 0 && reward[0] <= 1);
            }
        }
    }
}
