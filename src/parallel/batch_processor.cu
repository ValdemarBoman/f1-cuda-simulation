#include "batch_processor.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

__global__ void processBatchKernel(float* data, int batchSize, float* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize) {
        // Example processing: square the input data
        results[idx] = data[idx] * data[idx];
    }
}

void processBatch(float* data, int batchSize, float* results) {
    float *d_data, *d_results;
    
    cudaMalloc(&d_data, batchSize * sizeof(float));
    cudaMalloc(&d_results, batchSize * sizeof(float));
    
    cudaMemcpy(d_data, data, batchSize * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (batchSize + blockSize - 1) / blockSize;
    processBatchKernel<<<numBlocks, blockSize>>>(d_data, batchSize, d_results);
    
    cudaMemcpy(results, d_results, batchSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_data);
    cudaFree(d_results);
}