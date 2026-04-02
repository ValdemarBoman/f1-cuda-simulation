#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <iostream>

inline void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) 
                  << " in " << file << " at line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA_ERROR(err) (checkCudaError(err, __FILE__, __LINE__))

inline void allocateCudaMemory(void** devPtr, size_t size) {
    CHECK_CUDA_ERROR(cudaMalloc(devPtr, size));
}

inline void copyToDevice(void* devPtr, const void* hostPtr, size_t size) {
    CHECK_CUDA_ERROR(cudaMemcpy(devPtr, hostPtr, size, cudaMemcpyHostToDevice));
}

inline void copyToHost(void* hostPtr, const void* devPtr, size_t size) {
    CHECK_CUDA_ERROR(cudaMemcpy(hostPtr, devPtr, size, cudaMemcpyDeviceToHost));
}

inline void freeCudaMemory(void* devPtr) {
    CHECK_CUDA_ERROR(cudaFree(devPtr));
}

inline void synchronizeDevice() {
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

#endif // CUDA_UTILS_H