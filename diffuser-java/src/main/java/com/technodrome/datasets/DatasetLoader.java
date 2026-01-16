package com.technodrome.datasets;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.io.*;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Utility class for loading dataset from files.
 * Supports CSV and JSON formats.
 */
public class DatasetLoader {

    /**
     * Load dataset from a JSON file.
     * Expected format:
     * {
     *   "episodes": [
     *     {
     *       "observations": [[...], [...], ...],
     *       "actions": [[...], [...], ...],
     *       "rewards": [[...], [...], ...],
     *       "terminals": [[...], [...], ...],
     *       "timeouts": [[...], [...], ...]
     *     },
     *     ...
     *   ]
     * }
     */
    public static List<Map<String, float[][]>> loadFromJson(String filePath) throws IOException {
        Gson gson = new Gson();
        String content = Files.readString(Path.of(filePath));

        Type datasetType = new TypeToken<Map<String, List<Map<String, float[][]>>>>() {}.getType();
        Map<String, List<Map<String, float[][]>>> dataset = gson.fromJson(content, datasetType);

        return dataset.get("episodes");
    }

    /**
     * Load dataset from CSV files.
     * Expects separate files for each field:
     * - observations.csv
     * - actions.csv
     * - rewards.csv
     * - terminals.csv (optional)
     * - timeouts.csv (optional)
     * - episode_starts.csv (indices where new episodes start)
     */
    public static List<Map<String, float[][]>> loadFromCsv(String directory) throws IOException {
        Path dir = Path.of(directory);

        float[][] observations = loadCsvArray(dir.resolve("observations.csv").toString());
        float[][] actions = loadCsvArray(dir.resolve("actions.csv").toString());
        float[][] rewards = loadCsvArray(dir.resolve("rewards.csv").toString());

        float[][] terminals = null;
        Path terminalsPath = dir.resolve("terminals.csv");
        if (Files.exists(terminalsPath)) {
            terminals = loadCsvArray(terminalsPath.toString());
        }

        float[][] timeouts = null;
        Path timeoutsPath = dir.resolve("timeouts.csv");
        if (Files.exists(timeoutsPath)) {
            timeouts = loadCsvArray(timeoutsPath.toString());
        }

        // Load episode start indices
        int[] episodeStarts;
        Path episodeStartsPath = dir.resolve("episode_starts.csv");
        if (Files.exists(episodeStartsPath)) {
            float[][] startsArray = loadCsvArray(episodeStartsPath.toString());
            episodeStarts = new int[startsArray.length];
            for (int i = 0; i < startsArray.length; i++) {
                episodeStarts[i] = (int) startsArray[i][0];
            }
        } else {
            // Assume single episode
            episodeStarts = new int[]{0};
        }

        // Split into episodes
        List<Map<String, float[][]>> episodes = new ArrayList<>();
        for (int i = 0; i < episodeStarts.length; i++) {
            int start = episodeStarts[i];
            int end = (i < episodeStarts.length - 1) ? episodeStarts[i + 1] : observations.length;

            Map<String, float[][]> episode = new HashMap<>();
            episode.put("observations", Arrays.copyOfRange(observations, start, end));
            episode.put("actions", Arrays.copyOfRange(actions, start, end));
            episode.put("rewards", Arrays.copyOfRange(rewards, start, end));

            if (terminals != null) {
                episode.put("terminals", Arrays.copyOfRange(terminals, start, end));
            }
            if (timeouts != null) {
                episode.put("timeouts", Arrays.copyOfRange(timeouts, start, end));
            }

            episodes.add(episode);
        }

        return episodes;
    }

    /**
     * Load a 2D array from a CSV file.
     */
    private static float[][] loadCsvArray(String filePath) throws IOException {
        List<float[]> rows = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            boolean firstLine = true;

            while ((line = reader.readLine()) != null) {
                // Skip header if present
                if (firstLine && line.contains(",") && !isNumeric(line.split(",")[0].trim())) {
                    firstLine = false;
                    continue;
                }
                firstLine = false;

                String[] parts = line.split(",");
                float[] row = new float[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    row[i] = Float.parseFloat(parts[i].trim());
                }
                rows.add(row);
            }
        }

        return rows.toArray(new float[0][]);
    }

    private static boolean isNumeric(String str) {
        try {
            Float.parseFloat(str);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    /**
     * Create a sample dataset for testing purposes.
     */
    public static List<Map<String, float[][]>> createSampleDataset(int nEpisodes, int maxPathLength,
                                                                    int observationDim, int actionDim) {
        Random random = new Random(42);
        List<Map<String, float[][]>> episodes = new ArrayList<>();

        for (int ep = 0; ep < nEpisodes; ep++) {
            int pathLength = random.nextInt(maxPathLength / 2) + maxPathLength / 2;

            float[][] observations = new float[pathLength][observationDim];
            float[][] actions = new float[pathLength][actionDim];
            float[][] rewards = new float[pathLength][1];
            float[][] terminals = new float[pathLength][1];
            float[][] timeouts = new float[pathLength][1];

            for (int t = 0; t < pathLength; t++) {
                for (int d = 0; d < observationDim; d++) {
                    observations[t][d] = random.nextFloat() * 2 - 1;
                }
                for (int d = 0; d < actionDim; d++) {
                    actions[t][d] = random.nextFloat() * 2 - 1;
                }
                rewards[t][0] = random.nextFloat();
                terminals[t][0] = (t == pathLength - 1 && random.nextFloat() < 0.3) ? 1.0f : 0.0f;
                timeouts[t][0] = 0.0f;
            }

            Map<String, float[][]> episode = new HashMap<>();
            episode.put("observations", observations);
            episode.put("actions", actions);
            episode.put("rewards", rewards);
            episode.put("terminals", terminals);
            episode.put("timeouts", timeouts);

            episodes.add(episode);
        }

        return episodes;
    }
}
