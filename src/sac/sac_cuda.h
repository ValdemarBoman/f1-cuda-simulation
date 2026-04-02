#ifndef SAC_CUDA_H
#define SAC_CUDA_H

#include <vector>
#include "network.h"

struct Transition {
    std::vector<float> state;
    std::vector<float> action;
    float reward;
    std::vector<float> nextState;
    bool done;
};

struct SAC {
    // SAC parameters
    float alpha;
    int batchSize;
    int updatesPerStep;
    int learnStart;
    
    int obsDim, actionDim;
    
    // Neural networks on GPU
    NeuralNetwork policyNetwork;
    NeuralNetwork qNetwork1;
    NeuralNetwork qNetwork2;
    NeuralNetwork valueNetwork;
    NeuralNetwork targetValueNetwork;
    
    // Replay buffer
    std::vector<Transition> replayBuffer;
    int maxReplaySize;
    
    // Constructor
    SAC(int obsDim, int actionDim, unsigned long long seed);
    
    void storeTransition(const Transition& t);
    void update();
    void sampleBatch(std::vector<Transition>& batch);
    std::vector<float> act(const std::vector<float>& state, bool training = true);
    void train(const std::vector<Transition>& batch);
};

#endif // SAC_CUDA_H