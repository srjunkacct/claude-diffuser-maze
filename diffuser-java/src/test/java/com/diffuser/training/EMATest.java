package com.diffuser.training;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class EMATest {

    @Test
    void testEMACreation() {
        EMA ema = new EMA(0.995f);

        assertEquals(0.995f, ema.getBeta(), 0.0001f);
    }

    @Test
    void testEMABetaRange() {
        // Beta should typically be close to 1 for slow-moving average
        EMA ema = new EMA(0.999f);
        assertTrue(ema.getBeta() > 0.9f);
        assertTrue(ema.getBeta() < 1.0f);
    }
}
