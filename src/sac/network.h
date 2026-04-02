#ifndef NETWORK_H
#define NETWORK_H

#include <cuda_runtime.h>
#include <vector>

struct Layer {
    int inputSize;
    int outputSize;
    float* weights; // Pointer to weights in device memory
    float* biases;  // Pointer to biases in device memory

    Layer(int inSize, int outSize);
    ~Layer();
    void forward(float* input, float* output);
    void backward(float* input, float* output, float* d_output);
};

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layerSizes);
    ~NeuralNetwork();
    void forward(float* input, float* output);
    void backward(float* input, float* output, float* d_output);
    void updateWeights(float learningRate);

private:
    std::vector<Layer*> layers;
};

#endif // NETWORK_H