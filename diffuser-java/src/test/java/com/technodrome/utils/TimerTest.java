package com.technodrome.utils;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TimerTest {

    @Test
    void testElapsed() throws InterruptedException {
        Timer timer = new Timer();

        // Wait a bit
        Thread.sleep(100);

        double elapsed = timer.elapsed();

        // Should be approximately 0.1 seconds
        assertTrue(elapsed >= 0.09 && elapsed <= 0.2,
                "Elapsed time should be approximately 0.1 seconds, was: " + elapsed);
    }

    @Test
    void testReset() throws InterruptedException {
        Timer timer = new Timer();

        Thread.sleep(100);
        double elapsed1 = timer.elapsed(true);  // Reset

        Thread.sleep(50);
        double elapsed2 = timer.elapsed(true);

        // Second elapsed should be smaller (timer was reset)
        assertTrue(elapsed2 < elapsed1,
                "After reset, elapsed time should be smaller");
    }

    @Test
    void testNoReset() throws InterruptedException {
        Timer timer = new Timer();

        Thread.sleep(50);
        double elapsed1 = timer.elapsed(false);  // Don't reset

        Thread.sleep(50);
        double elapsed2 = timer.elapsed(false);

        // Second elapsed should be larger (timer not reset)
        assertTrue(elapsed2 > elapsed1,
                "Without reset, elapsed time should continue increasing");
    }
}
