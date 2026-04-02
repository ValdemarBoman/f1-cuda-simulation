#include "track.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

__device__ void loadTrackPoints(const char* path, trackPoint* points, int* numPoints) {
    std::ifstream in(path);
    if (!in) {
        printf("Failed to open '%s'. Make sure the file exists.\n", path);
        return;
    }

    std::string headerLine;
    if (!getline(in, headerLine)) {
        printf("CSV is empty\n");
        return;
    }

    std::string line;
    int idx = 0;
    while (getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string token;

        // Read x, y, wl, wr
        if (getline(ss, token, ',')) points[idx].ppos.x = std::stod(token);
        if (getline(ss, token, ',')) points[idx].ppos.y = std::stod(token);
        if (getline(ss, token, ',')) points[idx].wr = std::stod(token);
        if (getline(ss, token, ',')) points[idx].wl = std::stod(token);
        idx++;
    }
    *numPoints = idx;
}

__device__ double calculateTrackLength(const trackPoint* points, int numPoints) {
    double length = 0.0;
    for (int i = 1; i < numPoints; ++i) {
        double dx = points[i].ppos.x - points[i - 1].ppos.x;
        double dy = points[i].ppos.y - points[i - 1].ppos.y;
        length += sqrt(dx * dx + dy * dy);
    }
    return length;
}

__global__ void processTrackData(const char* path, trackPoint* points, int* numPoints, double* trackLength) {
    loadTrackPoints(path, points, numPoints);
    *trackLength = calculateTrackLength(points, *numPoints);
}

void manageTrack(const char* path, trackPoint* d_points, int* d_numPoints, double* d_trackLength) {
    processTrackData<<<1, 1>>>(path, d_points, d_numPoints, d_trackLength);
    cudaDeviceSynchronize();
}