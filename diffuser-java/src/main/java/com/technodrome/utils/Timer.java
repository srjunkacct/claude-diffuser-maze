package com.technodrome.utils;

/**
 * Simple timer for measuring elapsed time.
 */
public class Timer {

    private long startTime;

    public Timer() {
        this.startTime = System.currentTimeMillis();
    }

    /**
     * Get elapsed time in seconds and reset timer.
     */
    public double elapsed() {
        return elapsed(true);
    }

    /**
     * Get elapsed time in seconds.
     *
     * @param reset whether to reset the timer
     * @return elapsed time in seconds
     */
    public double elapsed(boolean reset) {
        long now = System.currentTimeMillis();
        double diff = (now - startTime) / 1000.0;
        if (reset) {
            startTime = now;
        }
        return diff;
    }

    /**
     * Reset the timer.
     */
    public void reset() {
        startTime = System.currentTimeMillis();
    }
}
