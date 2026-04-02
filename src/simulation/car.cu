#include "car.h"
#include "physics.h"
#include <cuda_runtime.h>

__device__ void updateCarDynamics(carState* car, const carParams* params, const physics* phy, double dt) {
    // Update car velocity based on acceleration
    car->vel.x += car->acc.x * dt;
    car->vel.y += car->acc.y * dt;

    // Update car position based on velocity
    car->pos.x += car->vel.x * dt;
    car->pos.y += car->vel.y * dt;

    // Reset acceleration for the next step
    car->acc.x = 0.0;
    car->acc.y = 0.0;
}

__global__ void simulateCars(carState* cars, const carParams* params, const physics* phy, int numCars, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numCars) {
        updateCarDynamics(&cars[idx], params, phy, dt);
    }
}

void launchCarSimulation(carState* d_cars, const carParams* d_params, const physics* d_phy, int numCars, double dt) {
    int blockSize = 256;
    int numBlocks = (numCars + blockSize - 1) / blockSize;
    simulateCars<<<numBlocks, blockSize>>>(d_cars, d_params, d_phy, numCars, dt);
    cudaDeviceSynchronize();
}