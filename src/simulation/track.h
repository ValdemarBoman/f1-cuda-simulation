#ifndef TRACK_H
#define TRACK_H

#include <vector>
#include "cuda_utils.h"

struct Vec2 {
    double x, y;

    __host__ __device__ Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    __host__ __device__ Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    __host__ __device__ Vec2 operator*(double s) const { return {x * s, y * s}; }
    __host__ __device__ Vec2 operator/(double s) const { return {x / s, y / s}; }

    __host__ __device__ Vec2& normalize() {
        double len = sqrt(x * x + y * y);
        if (len > 0) {
            x /= len;
            y /= len;
        }
        return *this;
    }

    __host__ __device__ double norm() const {
        return sqrt(x * x + y * y);
    }

    __host__ __device__ Vec2 rotate90() const {
        return { y, -x };
    }

    __host__ __device__ double dot(const Vec2& o) const {
        return x * o.x + y * o.y;
    }
};

struct trackPoint {
    Vec2 ppos; // Coordinates of centerline
    double wl, wr; // Left and right width from centerline
};

struct Track {
    std::vector<trackPoint> points;
    double length; // Track length in meters
    std::vector<Vec2> T; // Tangent vectors
    std::vector<Vec2> N; // Normal vectors
};

void loadTrack(const char* path, Track& track);
double getTrackLength(const Track& track);
void calculateTrackGeometry(Track& track);

#endif // TRACK_H