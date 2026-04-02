#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <cmath>
#include <cuda_runtime.h>

__host__ __device__ inline double dot(const double* a, const double* b, int size) {
    double result = 0.0;
    for (int i = 0; i < size; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

__host__ __device__ inline void normalize(double* vec, int size) {
    double length = 0.0;
    for (int i = 0; i < size; ++i) {
        length += vec[i] * vec[i];
    }
    length = sqrt(length);
    if (length > 0) {
        for (int i = 0; i < size; ++i) {
            vec[i] /= length;
        }
    }
}

__host__ __device__ inline void cross(const double* a, const double* b, double* result) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

#endif // MATH_UTILS_H