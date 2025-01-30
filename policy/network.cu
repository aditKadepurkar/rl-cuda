/**
 * network.cu
 * 
 * 
 */


#include "network.h"
#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>



MLP::MLP(const std::vector<int>& layers) : layerSizes(layers) {
    cudnnCreate(&cudnn);
    cublasCreate(&cublas);
    numLayers = layers.size() - 1;

    initializeLayers();
}

void MLP::initializeLayers() {
    curandGenerator_t randGen;
    curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(randGen, 1234ULL);

    for (int i = 0; i < numLayers; ++i) {
        int inputSize = layerSizes[i];
        int outputSize = layerSizes[i + 1];

        float *d_weights, *d_biases, *d_activations;
        cudaMalloc(&d_weights, inputSize * outputSize * sizeof(float));
        cudaMalloc(&d_biases, outputSize * sizeof(float));
        cudaMalloc(&d_activations, outputSize * sizeof(float));

        // Initialize weights and biases with random values
        curandGenerateUniform(randGen, d_weights, inputSize * outputSize);
        curandGenerateUniform(randGen, d_biases, outputSize);

        weights.push_back(d_weights);
        biases.push_back(d_biases);
        activations.push_back(d_activations);
    }

    curandDestroyGenerator(randGen);
}

void MLP::forward(float* input, float* output) {

    float* currInput = input;
    float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < numLayers; ++i) {
        int inputSize = layerSizes[i];
        int outputSize = layerSizes[i + 1];

        // Perform matrix multiplication: Y = W * X + B
        cublasSgemm(
            cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            outputSize, 1, inputSize,
            &alpha,
            weights[i], outputSize, // W: (outputSize, inputSize)
            currInput, inputSize,   // X: (inputSize, 1)
            &beta,
            activations[i], outputSize // Y: (outputSize, 1)
        );

        cublasSaxpy(cublas, outputSize, &alpha, biases[i], 1, activations[i], 1);

        currInput = activations[i];
    }

    cudaMemcpy(output, activations.back(), layerSizes.back() * sizeof(float), cudaMemcpyDeviceToHost);
}

MLP::~MLP() {
    cleanup();
    cudnnDestroy(cudnn);
    cublasDestroy(cublas);
}

void MLP::cleanup() {
    for (auto& w : weights) cudaFree(w);
    for (auto& b : biases) cudaFree(b);
    for (auto& a : activations) cudaFree(a);
}
