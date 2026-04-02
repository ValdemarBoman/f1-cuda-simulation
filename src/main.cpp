#include <iostream>
#include <cuda_runtime.h>
#include "simulation/physics.h"
#include "simulation/track.h"
#include "simulation/car.h"
#include "sac/sac_cuda.h"
#include "parallel/episode_manager.h"
#include "utils/cuda_utils.h"

void initializeCUDA() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        exit(EXIT_FAILURE);
    }
    cudaSetDevice(0);
}

int main() {
    initializeCUDA();

    // Initialize simulation parameters
    Track track;
    Car car;
    Physics physics;
    SAC sac;

    // Load track data
    loadTrackData(&track, "Silverstone.csv");

    // Initialize car state
    initializeCar(&car, track);

    // Main simulation loop
    const int maxEpisodes = 100;
    for (int episode = 0; episode < maxEpisodes; ++episode) {
        resetCar(&car);
        double totalReward = 0.0;

        // Run an episode
        while (!isEpisodeFinished(&car, track)) {
            // Perform physics calculations
            updatePhysics(&car, &physics);

            // Get observations for SAC
            double observations[OBS_DIM];
            getObservations(&car, observations);

            // Get action from SAC
            double action[2];
            sac.act(observations, action);

            // Apply action to car
            applyAction(&car, action);

            // Update total reward
            totalReward += calculateReward(&car, track);
        }

        std::cout << "Episode " << episode << " finished with total reward: " << totalReward << std::endl;
    }

    return 0;
}