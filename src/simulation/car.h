#ifndef CAR_H
#define CAR_H

#include "physics.h"
#include <cuda_runtime.h>

struct CarParams {
    double mass; // Car mass
    double hp;   // Horsepower
};

struct CarState {
    Vec2 pos;    // Position of the car
    Vec2 vel;    // Velocity of the car
    Vec2 acc;    // Acceleration of the car in local frame
    tireState tire; // Tire state for friction coefficients
};

// CUDA kernel for updating car state
__global__ void updateCarState(CarState* carStates, const CarParams* carParams, double dt, int numCars);

#endif // CAR_H