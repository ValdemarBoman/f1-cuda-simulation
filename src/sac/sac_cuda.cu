#include "sac_cuda.h"
#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>

SAC::SAC(int obs, int act, unsigned long long seed)
    : obsDim(obs), actionDim(act), alpha(0.2f), batchSize(128),
      updatesPerStep(1), learnStart(2000), maxReplaySize(100000),
      policyNetwork(obs, act, {256, 256}, seed),
      qNetwork1(obs + act, 1, {256, 256}, seed + 1),
      qNetwork2(obs + act, 1, {256, 256}, seed + 2),
      valueNetwork(obs, 1, {256, 256}, seed + 3),
      targetValueNetwork(obs, 1, {256, 256}, seed + 4) {
    
    replayBuffer.reserve(maxReplaySize);
}

void SAC::storeTransition(const Transition& t) {
    if (replayBuffer.size() >= maxReplaySize) {
        replayBuffer.erase(replayBuffer.begin());
    }
    replayBuffer.push_back(t);
}

void SAC::sampleBatch(std::vector<Transition>& batch) {
    batch.clear();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, replayBuffer.size() - 1);
    
    for (int i = 0; i < batchSize; ++i) {
        batch.push_back(replayBuffer[dis(gen)]);
    }
}

std::vector<float> SAC::act(const std::vector<float>& state, bool training) {
    std::vector<float> action;
    policyNetwork.forward(state, action);
    
    if (training) {
        // Add exploration noise (Gaussian)
        std::random_device rd;
        std::normal_distribution<> gauss(0.0, 0.1);
        std::mt19937 gen(rd());
        for (auto& a : action) {
            a += gauss(gen);
            a = std::max(-1.0f, std::min(1.0f, a));
        }
    }
    
    return action;
}

void SAC::train(const std::vector<Transition>& batch) {
    // Simplified SAC training loop
    // In practice, you'd implement:
    // 1. Compute Q-targets using target value network
    // 2. Update Q-networks
    // 3. Update policy network
    // 4. Update value network
    // 5. Update temperature parameter alpha
    
    // This is a placeholder for the core SAC algorithm
    for (const auto& t : batch) {
        std::vector<float> nextValue;
        targetValueNetwork.forward(t.nextState, nextValue);
        
        float target = t.reward;
        if (!t.done) {
            target += 0.99f * nextValue[0];  // gamma = 0.99
        }
        
        // Compute loss and backprop
        std::vector<float> grad = {target};
        policyNetwork.backward(grad);
    }
}

void SAC::update() {
    if (replayBuffer.size() < learnStart) return;
    
    std::vector<Transition> batch;
    for (int i = 0; i < updatesPerStep; ++i) {
        sampleBatch(batch);
        train(batch);
    }
}