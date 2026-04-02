#include "network.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void forward_pass_kernel(float* input, float* output, float* weights, float* biases, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[i * output_size + idx];
        }
        output[idx] = sum + biases[idx];
    }
}

__global__ void backward_pass_kernel(float* input, float* output, float* weights, float* biases, float* grad_output, float* grad_input, int input_size, int output_size, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size) {
        float sum = 0.0f;
        for (int i = 0; i < output_size; i++) {
            sum += grad_output[i] * weights[idx * output_size + i];
        }
        grad_input[idx] = sum;

        for (int i = 0; i < output_size; i++) {
            float grad_weight = input[idx] * grad_output[i];
            weights[idx * output_size + i] -= learning_rate * grad_weight;
        }
    }
}

void forward_pass(float* input, float* output, float* weights, float* biases, int input_size, int output_size) {
    float *d_input, *d_output, *d_weights, *d_biases;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMalloc(&d_weights, input_size * output_size * sizeof(float));
    cudaMalloc(&d_biases, output_size * sizeof(float));

    cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, output_size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (output_size + blockSize - 1) / blockSize;
    forward_pass_kernel<<<numBlocks, blockSize>>>(d_input, d_output, d_weights, d_biases, input_size, output_size);

    cudaMemcpy(output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weights);
    cudaFree(d_biases);
}

void backward_pass(float* input, float* output, float* weights, float* biases, float* grad_output, float* grad_input, int input_size, int output_size, float learning_rate) {
    float *d_input, *d_output, *d_weights, *d_biases, *d_grad_output, *d_grad_input;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMalloc(&d_weights, input_size * output_size * sizeof(float));
    cudaMalloc(&d_biases, output_size * sizeof(float));
    cudaMalloc(&d_grad_output, output_size * sizeof(float));
    cudaMalloc(&d_grad_input, input_size * sizeof(float));

    cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_output, grad_output, output_size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (input_size + blockSize - 1) / blockSize;
    backward_pass_kernel<<<numBlocks, blockSize>>>(d_input, d_output, d_weights, d_biases, d_grad_output, d_grad_input, input_size, output_size, learning_rate);

    cudaMemcpy(grad_input, d_grad_input, input_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
}