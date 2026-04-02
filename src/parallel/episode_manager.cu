#include "episode_manager.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void initializeEpisodes(int* episodeStates, int numEpisodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numEpisodes) {
        episodeStates[idx] = 0; // Initialize episode state to 0
    }
}

__global__ void updateEpisodeStates(int* episodeStates, float* rewards, int numEpisodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numEpisodes) {
        episodeStates[idx] += rewards[idx]; // Update episode state with rewards
    }
}

void launchInitializeEpisodes(int* episodeStates, int numEpisodes) {
    int blockSize = 256;
    int numBlocks = (numEpisodes + blockSize - 1) / blockSize;
    initializeEpisodes<<<numBlocks, blockSize>>>(episodeStates, numEpisodes);
    cudaDeviceSynchronize();
}

void launchUpdateEpisodeStates(int* episodeStates, float* rewards, int numEpisodes) {
    int blockSize = 256;
    int numBlocks = (numEpisodes + blockSize - 1) / blockSize;
    updateEpisodeStates<<<numBlocks, blockSize>>>(episodeStates, rewards, numEpisodes);
    cudaDeviceSynchronize();
}