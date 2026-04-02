#ifndef PHYSICS_H
#define PHYSICS_H

#include <cmath>
#include "car.h"

struct PhysicsConstants {
    static constexpr double GRAVITY = 9.81; // Gravitational acceleration (m/s^2)
    static constexpr double AIR_DENSITY = 1.225; // Air density at sea level (kg/m^3)
    static constexpr double LIFT_COEFFICIENT = 5.0; // Lift coefficient (dimensionless)
};

__device__ double calculateWeight(double mass) {
    return mass * PhysicsConstants::GRAVITY;
}

__device__ double calculateDownforce(double velocity, double area = 1.0) {
    return 0.5 * PhysicsConstants::AIR_DENSITY * PhysicsConstants::LIFT_COEFFICIENT * area * velocity * velocity;
}

__device__ double calculateNormalForce(const CarState& car, double mass) {
    double weight = calculateWeight(mass);
    double downforce = calculateDownforce(car.velocity.norm());
    return weight + downforce;
}

#endif // PHYSICS_H