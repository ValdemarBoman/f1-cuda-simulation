#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <iostream>

void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

void* allocateCudaMemory(size_t size) {
    void* d_ptr;
    checkCudaError(cudaMalloc(&d_ptr, size), __FILE__, __LINE__);
    return d_ptr;
}

void freeCudaMemory(void* d_ptr) {
    checkCudaError(cudaFree(d_ptr), __FILE__, __LINE__);
}

void copyToDevice(void* d_ptr, const void* h_ptr, size_t size) {
    checkCudaError(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice), __FILE__, __LINE__);
}

void copyToHost(void* h_ptr, const void* d_ptr, size_t size) {
    checkCudaError(cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
}

void synchronizeDevice() {
    checkCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
}