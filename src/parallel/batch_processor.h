#ifndef BATCH_PROCESSOR_H
#define BATCH_PROCESSOR_H

#include <vector>
#include "sac.h"
#include "cuda_utils.h"

struct Batch {
    std::vector<double> states;
    std::vector<double> actions;
    std::vector<double> rewards;
    std::vector<double> next_states;
    std::vector<bool> dones;
};

class BatchProcessor {
public:
    BatchProcessor(size_t batch_size, size_t buffer_size);
    ~BatchProcessor();

    void addTransition(const std::vector<double>& state, const std::vector<double>& action, 
                      double reward, const std::vector<double>& next_state, bool done);
    void sampleBatch(Batch& batch);
    void clearBuffer();

private:
    size_t batch_size;
    size_t buffer_size;
    size_t current_size;

    std::vector<double> states_buffer;
    std::vector<double> actions_buffer;
    std::vector<double> rewards_buffer;
    std::vector<double> next_states_buffer;
    std::vector<bool> dones_buffer;

    void allocateBuffers();
};

#endif // BATCH_PROCESSOR_H