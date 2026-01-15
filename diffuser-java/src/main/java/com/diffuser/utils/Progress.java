package com.diffuser.utils;

import java.util.Map;

/**
 * Progress bar for tracking iteration progress.
 */
public class Progress {

    private final int total;
    private final String name;
    private final int ncol;
    private final int maxLength;
    private final int speedUpdateFreq;

    private int step;
    private long time0;
    private int step0;
    private double speed;

    public Progress(int total) {
        this(total, "Progress", 3, 20, 100);
    }

    public Progress(int total, String name, int ncol, int maxLength, int speedUpdateFreq) {
        this.total = total;
        this.name = name;
        this.ncol = ncol;
        this.maxLength = maxLength;
        this.speedUpdateFreq = speedUpdateFreq;

        this.step = 0;
        this.time0 = System.currentTimeMillis();
        this.step0 = 0;
        this.speed = 0;
    }

    /**
     * Update progress with parameters.
     */
    public void update(Map<String, Object> params) {
        update(params, 1);
    }

    /**
     * Update progress with parameters and custom step increment.
     */
    public void update(Map<String, Object> params, int n) {
        step += n;

        if (step % speedUpdateFreq == 0) {
            time0 = System.currentTimeMillis();
            step0 = step;
        }

        setDescription(params);
    }

    /**
     * Set description and print progress.
     */
    public void setDescription(Map<String, Object> params) {
        String percentStr = formatPercent(step, total);
        String speedStr = formatSpeed(step);

        StringBuilder paramsStr = new StringBuilder();
        if (params != null && !params.isEmpty()) {
            for (Map.Entry<String, Object> entry : params.entrySet()) {
                if (paramsStr.length() > 0) paramsStr.append(" | ");
                paramsStr.append(entry.getKey()).append(": ").append(entry.getValue());
            }
        }

        String description = String.format("%s | %s | %s", percentStr, speedStr, paramsStr);
        System.out.print("\r" + description);
    }

    private String formatPercent(int n, int total) {
        if (total > 0) {
            double percent = (double) n / total;
            int pbarSize = ncol * maxLength;
            int completeEntries = (int) (percent * pbarSize);
            int incompleteEntries = pbarSize - completeEntries;

            String pbar = "#".repeat(Math.max(0, completeEntries)) +
                         " ".repeat(Math.max(0, incompleteEntries));

            String fraction = String.format("%d / %d", n, total);
            return String.format("%s [%s] %3d%%", fraction, pbar, (int) (percent * 100));
        } else {
            return String.format("%d iterations", n);
        }
    }

    private String formatSpeed(int n) {
        int numSteps = n - step0;
        double t = (System.currentTimeMillis() - time0) / 1000.0;
        if (t > 0 && numSteps > 0) {
            speed = numSteps / t;
        }
        return String.format("%.1f Hz", speed);
    }

    /**
     * Print final status and newline.
     */
    public void stamp() {
        System.out.println();
    }

    /**
     * Close the progress bar.
     */
    public void close() {
        System.out.println();
    }

    public int getStep() {
        return step;
    }

    public int getTotal() {
        return total;
    }
}

/**
 * Silent progress bar that does nothing.
 */
class Silent {
    public void update(Map<String, Object> params) {}
    public void update(Map<String, Object> params, int n) {}
    public void setDescription(Map<String, Object> params) {}
    public void stamp() {}
    public void close() {}
}
