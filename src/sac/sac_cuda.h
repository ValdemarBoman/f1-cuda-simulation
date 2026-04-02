#ifndef SAC_CUDA_H
#define SAC_CUDA_H

#include <vector>
#include "network.h"

struct SAC {
    // SAC parameters
    float alpha; // Temperature parameter
    int batchSize;
    int updatesPerStep;

    // Neural networks for policy and value functions
    NeuralNetwork policyNetwork;
    NeuralNetwork valueNetwork;

    // Replay buffer for storing transitions
    std::vector<Transition> replayBuffer;

    // Constructor
    SAC(int obsDim, int actionDim, float alpha, int batchSize, int updatesPerStep);

    // Function to store transitions in the replay buffer
    void storeTransition(const Transition& transition);

    // Function to update the policy and value networks
    void update();

    // Function to sample a batch from the replay buffer
    void sampleBatch(std::vector<Transition>& batch);

    // Function to select an action based on the current state
    std::vector<float> act(const std::vector<float>& state, bool training);

    // Function to perform a training step
    void train(const std::vector<Transition>& batch);
};

#endif // SAC_CUDA_H