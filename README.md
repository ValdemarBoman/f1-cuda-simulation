# F1 Simulation - Python + CUDA

Direct Python port of the F1 racing simulation with SAC reinforcement learning agent. Optimized for CUDA/GPU execution on Spark DGX and other NVIDIA systems.

## Features

- **F1 Physics Simulation**: Realistic car dynamics with friction, downforce, power limits
- **Soft Actor-Critic (SAC)**: Deep RL algorithm for autonomous racing agent training
- **CUDA Support**: Automatic GPU acceleration using PyTorch (falls back to CPU)
- **Track Data**: Silverstone circuit included
- **Comprehensive Logging**: Episode metrics, step-by-step data, and trajectory files

## Installation

### Requirements
- Python 3.8+
- NVIDIA CUDA Toolkit 11.8+ (optional, for GPU acceleration)
- cuDNN (optional, for GPU)

### Setup

```bash
# Clone repository
git clone <repo-url>
cd simulationF1-python

# Install dependencies
pip install -r requirements.txt
```

For GPU acceleration, ensure PyTorch is built with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Running

```bash
python3 main.py
```

The simulation will:
1. Load the Silverstone track
2. Run 100 training episodes (configurable in main.py)
3. Output CSV logs to `logs/` directory
4. Display progress to stdout

## Configuration

Edit `main.py` to adjust:
- `EPISODES`: Number of training episodes (default: 100)
- `MAX_STEPS`: Max steps per episode (default: 6000)
- `OBS_DIM`: Observation space dimension (default: 8)
- SAC hyperparameters: `alpha`, `gamma`, `tau`, `batch_size`, `learn_start`

## Output

Generated files in `logs/`:
- `metrics_episode.csv`: Per-episode statistics
- `metrics_step.csv`: Per-step detailed data
- `episode_N_traj.csv`: Car trajectory for episode N
- `track_centerline.csv`: Track geometry

## Code Structure

- `main.py`: Simulation loop, physics engine, reward shaping
- `sac.py`: SAC algorithm implementation with PyTorch
- `Silverstone.csv`: Track definition file

## Performance

On CPU (MacBook Air): ~1 episode per 60-90 seconds
On GPU (NVIDIA DGX): Expected 5-10x speedup

## Notes

- The code is a direct port from C++ with no simplifications
- All reward shaping and physics calculations are identical to original
- NaN safety checks included for numerical stability
- Replay buffer capacity: 200,000 transitions
- Neural networks: Actor (128-128-16), Critic (256-256-1)

## License

Same as original C++ version
