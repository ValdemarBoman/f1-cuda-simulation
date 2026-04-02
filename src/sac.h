#ifndef SAC_H
#define SAC_H

#include <vector>
#include <cuda_runtime.h>

struct SACParams {
    int obs_dim; // Observation dimension
    int action_dim; // Action dimension
    int replay_buffer_size; // Size of the replay buffer
    float learning_rate; // Learning rate for the optimizer
    float discount_factor; // Discount factor for future rewards
    float tau; // Soft update parameter
};

struct Transition {
    float* obs; // Observation
    float* action; // Action taken
    float reward; // Reward received
    float* next_obs; // Next observation
    bool done; // Whether the episode has ended
};

class SAC {
public:
    SAC(const SACParams& params);
    ~SAC();

    void storeTransition(const Transition& transition);
    void updatePolicy();
    void updateValueFunction();
    void sampleBatch(std::vector<Transition>& batch);
    
    void setDevice(int device_id);
    void synchronize();

private:
    SACParams params;
    float* d_replay_buffer; // Device pointer for replay buffer
    int replay_index; // Current index in the replay buffer
    int batch_size; // Size of the batch for updates
};

#endif // SAC_H