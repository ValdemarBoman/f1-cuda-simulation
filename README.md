# F1 CUDA Simulation

GPU-accelerated F1 racing simulation using NVIDIA CUDA and Soft Actor-Critic (SAC) reinforcement learning.

## Requirements

- **NVIDIA GPU** (A100 for DGX recommended, sm_80 architecture)
- CUDA 11.8+
- cuBLAS, cuRAND
- CMake 3.17+
- C++17 compiler

## Building on DGX

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
make -j8
./f1_simulation
```

## Building on Mac (CPU-only)

```bash
brew install cmake
mkdir build && cd build
cmake .. -DENABLE_CUDA=OFF
make -j8
./f1_simulation
```

## Project Structure

```
src/
├── main.cpp           # Main simulation loop
├── sac/
│   ├── network.h      # GPU neural network layers
│   ├── network.cu     # CUDA kernels
│   ├── sac_cuda.h     # SAC algorithm header
│   └── sac_cuda.cu    # SAC training implementation
└── utils/
    └── cuda_utils.h   # CUDA utility functions
```

## Features

- ✅ Parallel episode simulation on GPU
- ✅ cuBLAS-accelerated neural networks
- ✅ Soft Actor-Critic (SAC) training
- ✅ DGX A100 optimized

## License

MIT