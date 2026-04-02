#include "network.h"
#include <stdio.h>
#include <math.h>

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

#define CUBLAS_CHECK(call) \
  do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
      fprintf(stderr, "cuBLAS error: %d\n", status); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void relu_backward_kernel(float* gradOut, float* input, float* gradIn, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gradIn[idx] = (input[idx] > 0.0f) ? gradOut[idx] : 0.0f;
    }
}

__global__ void add_bias_kernel(float* output, float* bias, int batchSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize * outputSize) {
        int batch = idx / outputSize;
        int feat = idx % outputSize;
        output[idx] += bias[feat];
    }
}

Layer::Layer(int in, int out, cublasHandle_t handle)
    : inputSize(in), outputSize(out), learningRate(1e-4) {
    
    CUDA_CHECK(cudaMalloc(&d_weights, in * out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dWeights, in * out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dBias, out * sizeof(float)));
    
    CUDA_CHECK(cudaMalloc(&d_input, in * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, out * sizeof(float)));
    
    CUDA_CHECK(cudaMemset(d_weights, 0, in * out * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_bias, 0, out * sizeof(float)));
}

Layer::~Layer() {
    if (d_weights) cudaFree(d_weights);
    if (d_bias) cudaFree(d_bias);
    if (d_dWeights) cudaFree(d_dWeights);
    if (d_dBias) cudaFree(d_dBias);
    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);
}

void Layer::initializeWeights(curandGenerator_t gen) {
    curandGenerateNormal(gen, d_weights, inputSize * outputSize, 0.0f, 0.01f);
    curandGenerateNormal(gen, d_bias, outputSize, 0.0f, 0.001f);
}

void Layer::forward(float* input, cublasHandle_t handle) {
    // Copy input to device memory
    CUDA_CHECK(cudaMemcpy(d_input, input, inputSize * sizeof(float), cudaMemcpyHostToDevice));
    
    // Compute: output = weights^T * input + bias
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, 
                             inputSize, outputSize,
                             &alpha, d_weights, inputSize,
                             d_input, 1,
                             &beta, d_output, 1));
    
    // Add bias
    int blockSize = 256;
    int gridSize = (outputSize + blockSize - 1) / blockSize;
    add_bias_kernel<<<gridSize, blockSize>>>(d_output, d_bias, 1, outputSize);
    
    // ReLU activation (except last layer)
    blockSize = 256;
    gridSize = (outputSize + blockSize - 1) / blockSize;
    relu_kernel<<<gridSize, blockSize>>>(d_output, outputSize);
}

void Layer::backward(float* gradOutput, cublasHandle_t handle) {
    // Compute gradient w.r.t. weights: dW = input * gradOutput^T
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSger(handle, inputSize, outputSize,
                            &alpha, d_input, 1,
                            gradOutput, 1,
                            d_dWeights, inputSize));
    
    // Compute gradient w.r.t. bias: dB = sum(gradOutput)
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N,
                             outputSize, 1,
                             &alpha, gradOutput, outputSize,
                             nullptr, 1,
                             &beta, d_dBias, 1));
}

void Layer::updateWeights(cublasHandle_t handle) {
    float alpha = -learningRate;
    CUBLAS_CHECK(cublasSaxpy(handle, inputSize * outputSize,
                             &alpha, d_dWeights, 1,
                             d_weights, 1));
    
    CUBLAS_CHECK(cublasSaxpy(handle, outputSize,
                             &alpha, d_dBias, 1,
                             d_bias, 1));
}

NeuralNetwork::NeuralNetwork(int inDim, int outDim, const std::vector<int>& hidden, unsigned long long seed)
    : inputDim(inDim), outputDim(outDim), hiddenLayers(hidden) {
    
    CUBLAS_CHECK(cublasCreate(&cublasHandle));
    curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curandGen, seed);
    
    std::vector<int> layerSizes = {inDim};
    layerSizes.insert(layerSizes.end(), hidden.begin(), hidden.end());
    layerSizes.push_back(outDim);
    
    for (size_t i = 0; i < layerSizes.size() - 1; ++i) {
        layers.emplace_back(layerSizes[i], layerSizes[i+1], cublasHandle);
        layers.back().initializeWeights(curandGen);
    }
}

NeuralNetwork::~NeuralNetwork() {
    if (cublasHandle) cublasDestroy(cublasHandle);
    if (curandGen) curandDestroyGenerator(curandGen);
}

void NeuralNetwork::forward(const std::vector<float>& input, std::vector<float>& output) {
    if (input.size() != inputDim) {
        fprintf(stderr, "Input size mismatch: %zu != %d\n", input.size(), inputDim);
        return;
    }
    
    std::vector<float> current = input;
    
    for (size_t i = 0; i < layers.size(); ++i) {
        layers[i].forward(current.data(), cublasHandle);
        
        current.resize(layers[i].outputSize);
        CUDA_CHECK(cudaMemcpy(current.data(), layers[i].d_output, 
                              layers[i].outputSize * sizeof(float), 
                              cudaMemcpyDeviceToHost));
    }
    
    output = current;
}

void NeuralNetwork::backward(const std::vector<float>& gradOutput) {
    std::vector<float> grad = gradOutput;
    
    for (int i = (int)layers.size() - 1; i >= 0; --i) {
        layers[i].backward(grad.data(), cublasHandle);
        layers[i].updateWeights(0.0001f);  // learning rate
        
        if (i > 0) {
            grad.resize(layers[i-1].outputSize);
            // Backprop through weights
            float alpha = 1.0f, beta = 0.0f;
            CUBLAS_CHECK(cublasSgemv(cublasHandle, CUBLAS_OP_N,
                                     layers[i].inputSize, layers[i].outputSize,
                                     &alpha, layers[i].d_weights, layers[i].inputSize,
                                     grad.data(), 1,
                                     &beta, grad.data(), 1));
        }
    }
}

void NeuralNetwork::updateWeights(float learningRate) {
    for (auto& layer : layers) {
        layer.learningRate = learningRate;
        layer.updateWeights(cublasHandle);
    }
}

void NeuralNetwork::copyWeightsFromHost(const std::vector<float>& weights) {
    int offset = 0;
    for (auto& layer : layers) {
        int wSize = layer.inputSize * layer.outputSize;
        int bSize = layer.outputSize;
        
        CUDA_CHECK(cudaMemcpy(layer.d_weights, weights.data() + offset, 
                              wSize * sizeof(float), cudaMemcpyHostToDevice));
        offset += wSize;
        
        CUDA_CHECK(cudaMemcpy(layer.d_bias, weights.data() + offset,
                              bSize * sizeof(float), cudaMemcpyHostToDevice));
        offset += bSize;
    }
}

void NeuralNetwork::copyWeightsToHost(std::vector<float>& weights) {
    weights.clear();
    for (const auto& layer : layers) {
        std::vector<float> w(layer.inputSize * layer.outputSize);
        std::vector<float> b(layer.outputSize);
        
        CUDA_CHECK(cudaMemcpy(w.data(), layer.d_weights, 
                              w.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), layer.d_bias,
                              b.size() * sizeof(float), cudaMemcpyDeviceToHost));
        
        weights.insert(weights.end(), w.begin(), w.end());
        weights.insert(weights.end(), b.begin(), b.end());
    }
}