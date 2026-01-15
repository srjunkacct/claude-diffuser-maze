# Claude Code Guide for Diffuser Project

## Project Overview

This repository contains a **Diffusion Model for Offline Reinforcement Learning** implementation. The project exists in two implementations:
- **Python (PyTorch)**: Original implementation in `diffuser/`
- **Java (DJL)**: Port using Deep Java Library in `diffuser-java/`

The diffusion model learns to generate trajectories by iteratively denoising random noise, enabling planning and decision-making in RL environments.

## Build Commands

### Java (Maven)
```bash
cd diffuser-java

# Compile
mvn compile

# Run tests
mvn test

# Run specific test class
mvn test -Dtest=GaussianDiffusionTest

# Package
mvn package
```

### Python
```bash
cd diffuser

# Install dependencies
pip install -e .

# Run training (example)
python scripts/train.py --dataset maze2d-umaze-v1
```

## Architecture

### Core Components

```
GaussianDiffusion (Main Model)
├── TemporalUnet (Denoising Network)
│   ├── ResidualTemporalBlock (with skip connections)
│   ├── SinusoidalPosEmb (timestep encoding)
│   └── Conv1dBlock (BatchNorm + Mish activation)
├── Beta Schedule (cosine schedule for noise variance)
└── Loss Functions (WeightedLoss, ValueLoss)

Dataset Pipeline
├── SequenceDataset (trajectory sequences)
│   ├── ReplayBuffer (episode storage)
│   └── Normalizers (Limits, Gaussian, SafeLimits)
├── GoalDataset (goal-conditioned)
└── ValueDataset (with value targets)

Training
├── DiffusionTrainer (training loop)
├── EMA (exponential moving average)
└── Policy (inference wrapper)
```

### Key Concepts

1. **Forward Process**: Gradually adds Gaussian noise to data over T timesteps
2. **Reverse Process**: Learns to denoise, generating clean trajectories from noise
3. **Cosine Beta Schedule**: Controls noise variance at each timestep
4. **Temporal U-Net**: 1D convolutional network with skip connections for denoising
5. **EMA**: Maintains smoothed model weights for stable inference

## Directory Structure

### Java (`diffuser-java/`)
```
src/main/java/com/diffuser/
├── models/
│   ├── GaussianDiffusion.java      # Main diffusion model
│   ├── TemporalUnet.java           # Denoising U-Net
│   ├── TemporalValue.java          # Value estimation network
│   ├── ResidualTemporalBlock.java  # Residual blocks
│   └── helpers/                    # Conv1dBlock, Mish, etc.
├── datasets/
│   ├── SequenceDataset.java        # Main dataset class
│   ├── ReplayBuffer.java           # Episode storage
│   ├── DatasetLoader.java          # File loading utilities
│   └── normalization/              # Normalizer implementations
├── training/
│   ├── DiffusionTrainer.java       # Training loop
│   └── EMA.java                    # Exponential moving average
├── policy/
│   └── Policy.java                 # Inference wrapper
└── utils/                          # Timer, ArrayUtils, etc.
```

### Python (`diffuser/`)
```
diffuser/
├── models/
│   ├── diffusion.py                # GaussianDiffusion
│   ├── temporal.py                 # TemporalUnet, TemporalValue
│   └── helpers.py                  # Conv1d, SinusoidalPosEmb
├── datasets/
│   ├── sequence.py                 # SequenceDataset
│   ├── buffer.py                   # ReplayBuffer
│   └── normalization.py            # Normalizers
└── utils/
    ├── training.py                 # Trainer, EMA
    └── arrays.py                   # Array utilities
```

## Key Implementation Details

### DJL-Specific Notes (Java)
- Uses **DJL 0.31.0** with PyTorch backend
- **BatchNorm** instead of GroupNorm (GroupNorm not available in DJL)
- Parameter access via `getParameters().valueAt(i)` indexed iteration
- Gradient collection via `Engine.getInstance().newGradientCollector()`
- Dataset extends `RandomAccessDataset` with `prepare()` and `get()` methods
- PyTorch native libraries auto-downloaded by DJL at runtime

### Model Dimensions
- `transitionDim`: action_dim + observation_dim (trajectory dimension)
- `horizon`: sequence length for planning
- `dim`: base channel dimension (typically 32 or 64)
- `dimMults`: channel multipliers for U-Net levels [1, 2, 4, 8]

### Training Configuration
- Learning rate: 2e-5
- EMA decay: 0.995
- Gradient accumulation: 2 steps
- Batch size: 32
- Diffusion timesteps: 200 (default)

## Testing

Java tests are in `src/test/java/com/diffuser/`:
- Unit tests for each component (models, datasets, training)
- All 28 tests should pass with `mvn test`

## Common Tasks

### Adding a New Normalizer
1. Create class in `datasets/normalization/` implementing `Normalizer` interface
2. Add case to `createNormalizer()` in `SequenceDataset.java`

### Modifying Network Architecture
1. Edit `TemporalUnet.java` for denoising network changes
2. Adjust `dimMults` array for different depth
3. Update `getOutputShapes()` if output dimensions change

### Changing Diffusion Schedule
1. Modify `cosine_beta_schedule()` in `DiffusionHelpers.java`
2. Adjust `nTimesteps` in `GaussianDiffusion` constructor
