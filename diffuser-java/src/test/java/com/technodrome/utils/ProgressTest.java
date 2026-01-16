package com.technodrome.utils;

import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ProgressTest {

    @Test
    void testProgressCreation() {
        Progress progress = new Progress(100);

        assertEquals(0, progress.getStep());
        assertEquals(100, progress.getTotal());
    }

    @Test
    void testProgressUpdate() {
        Progress progress = new Progress(100);

        Map<String, Object> params = new HashMap<>();
        params.put("loss", 0.5f);
        params.put("step", 10);

        progress.update(params);

        assertEquals(1, progress.getStep());
    }

    @Test
    void testProgressMultipleUpdates() {
        Progress progress = new Progress(100);

        for (int i = 0; i < 50; i++) {
            progress.update(new HashMap<>());
        }

        assertEquals(50, progress.getStep());
    }

    @Test
    void testProgressCustomIncrement() {
        Progress progress = new Progress(100);

        progress.update(new HashMap<>(), 5);
        assertEquals(5, progress.getStep());

        progress.update(new HashMap<>(), 10);
        assertEquals(15, progress.getStep());
    }
}
