#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

struct Layer {
    float* d_weights;
    float* d_bias;
    float* d_dWeights;  // gradients
    float* d_dBias;
    
    float* d_input;
    float* d_output;
    
    int inputSize, outputSize;
    float learningRate;

    Layer(int in, int out, cublasHandle_t handle);
    ~Layer();
    
    void forward(float* input, cublasHandle_t handle);
    void backward(float* gradOutput, cublasHandle_t handle);
    void updateWeights(cublasHandle_t handle);
    void initializeWeights(curandGenerator_t gen);
};

struct NeuralNetwork {
    std::vector<Layer> layers;
    int inputDim, outputDim;
    cublasHandle_t cublasHandle;
    curandGenerator_t curandGen;
    
    std::vector<int> hiddenLayers;

    NeuralNetwork(int inDim, int outDim, const std::vector<int>& hidden, unsigned long long seed = 1234);
    ~NeuralNetwork();
    
    void forward(const std::vector<float>& input, std::vector<float>& output);
    void backward(const std::vector<float>& gradOutput);
    void updateWeights(float learningRate);
    void copyWeightsFromHost(const std::vector<float>& weights);
    void copyWeightsToHost(std::vector<float>& weights);
};

#endif // NETWORK_H