#include "physics.h"
#include <cuda_runtime.h>

__device__ const double g = 9.81; // Gravitational acceleration (m/s^2)

__device__ void calculateForces(double mass, double velocity, double* force) {
    // Calculate gravitational force
    force[0] = mass * g; // Weight force
    // Additional forces can be calculated here (e.g., drag, downforce)
}

__device__ void updateMotion(double* position, double* velocity, double* acceleration, double dt) {
    // Update velocity and position based on acceleration
    velocity[0] += acceleration[0] * dt;
    velocity[1] += acceleration[1] * dt;

    position[0] += velocity[0] * dt;
    position[1] += velocity[1] * dt;
}

__global__ void simulatePhysics(double* positions, double* velocities, double* accelerations, double mass, double dt, int numCars) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numCars) {
        double force[2] = {0.0, 0.0};
        calculateForces(mass, velocities[idx * 2], force);
        
        // Update motion
        updateMotion(&positions[idx * 2], &velocities[idx * 2], force, dt);
    }
}

void launchPhysicsSimulation(double* positions, double* velocities, double* accelerations, double mass, double dt, int numCars) {
    int blockSize = 256;
    int numBlocks = (numCars + blockSize - 1) / blockSize;
    simulatePhysics<<<numBlocks, blockSize>>>(positions, velocities, accelerations, mass, dt, numCars);
    cudaDeviceSynchronize();
}