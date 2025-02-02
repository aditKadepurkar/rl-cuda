/**
 * network.cu
 * 
 * 
 */


#include "network.h"
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

// basic gradient descent update
__global__ void update_kernel(float* param, const float* grad, float learning_rate, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        param[idx] -= learning_rate * grad[idx];
    }
}

__global__ void activation_deriv_kernel(const float* activation, float* grad, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // relu
        grad[idx] *= (activation[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

__global__ void compute_value_grad_kernel(const float* predictions, const float* targets, float* grads, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // literally just MSE
        grads[idx] = 2.0f * (predictions[idx] - targets[idx]) / n;
    }
}

__global__ void log_prob_kernel(const float* means, const float* actions, float* log_probs, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float diff = actions[idx] - means[idx];
        
        const float epsilon = 1e-6f;
        float squared_diff = diff * diff + epsilon;

        float constant = 0.5f * logf(2.0f * 3.14159265f);
        log_probs[idx] = -0.5f * squared_diff - constant;
    }
}

__global__ void ratio_kernel(const float* new_log_probs, const float* old_log_probs, float* ratios, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        ratios[idx] = expf(new_log_probs[idx] - old_log_probs[idx]);
    }
}

__global__ void surrogate_loss_kernel(const float* ratios, const float* advantages, float* surrogate, int n, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float r = ratios[idx];
        float adv = advantages[idx];
        float r_clipped = fminf(fmaxf(r, 1.0f - epsilon), 1.0f + epsilon);
        float s1 = r * adv;
        float s2 = r_clipped * adv;
        surrogate[idx] = (s1 < s2) ? s1 : s2;
    }
}



// divide

MLP::MLP(const std::vector<int>& layers) : layerSizes(layers) {
    cudnnCreate(&cudnn);
    cublasCreate(&cublas);
    numLayers = layers.size() - 1;

    this->output_dim = layers.back();

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
        if (cudaMalloc(&d_weights, inputSize * outputSize * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&d_biases, outputSize * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&d_activations, outputSize * sizeof(float)) != cudaSuccess) {
            std::cerr << "CUDA malloc failed!" << std::endl;
            exit(1);
        }

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

    this->d_input = input;

    for (int i = 0; i < numLayers; ++i) {
        int inputSize = layerSizes[i];
        int outputSize = layerSizes[i + 1];

        // Perform matrix multiplication: Y = W * X + B
        cublasStatus_t status = cublasSgemm(
            cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            outputSize, 1, inputSize,
            &alpha,
            weights[i], outputSize, // W: (outputSize, inputSize)
            currInput, inputSize,   // X: (inputSize, 1)
            &beta,
            activations[i], outputSize // Y: (outputSize, 1)
        );

        // Check for errors in cublasSgemm
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublasSgemm failed with error code " << status << std::endl;
            exit(1);
        }

        cublasSaxpy(cublas, outputSize, &alpha, biases[i], 1, activations[i], 1);

        // Check if activations contain NaN after bias addition
        checkForNaN(activations[i], outputSize);

        currInput = activations[i];
    }

    checkForNaN(activations.back(), layerSizes.back());
    cudaMemcpy(output, activations.back(), layerSizes.back() * sizeof(float), cudaMemcpyDeviceToDevice);
}

// Helper function to check for NaN values in a CUDA array
void MLP::checkForNaN(float* data, int size) {
    float* hostData = new float[size];
    cudaMemcpy(hostData, data, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        if (std::isnan(hostData[i])) {
            std::cerr << "NaN detected in activation at index " << i << std::endl;
            exit(1);
        }
    }

    delete[] hostData;
}

// In network.cu

void MLP::backward(float* d_output_grad, float learning_rate) {
    // wtf did i cook up here (it is literally just backprop)

    // we ahve activations[i] for each layer


    float alpha = 1.0f, beta = 0.0f; // more hardcoded hyperparameters you love to see it TODO at sm pt


    float* d_delta = d_output_grad;  // wthis should have shape of  [layerSizes.back()]

    // backprop
    for (int layer = numLayers - 1; layer >= 0; --layer) {
        int inputSize = layerSizes[layer];
        int outputSize = layerSizes[layer + 1];

        float* d_prev_activation = (layer == 0) ? d_input : activations[layer - 1];

        float* d_dW;
        cudaMalloc(&d_dW, inputSize * outputSize * sizeof(float));

        // why was this so hard to get working
        cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                    outputSize, inputSize, 1,
                    &alpha,
                    d_delta, outputSize,
                    d_prev_activation, inputSize,
                    &beta,
                    d_dW, outputSize);

        // updating weights and biases!! (wandb reference??)
        int numW = inputSize * outputSize;
        int threads = 256;
        int blocks = (numW + threads - 1) / threads;
        update_kernel<<<blocks, threads>>>(weights[layer], d_dW, learning_rate, numW);
        cudaDeviceSynchronize();
        cudaFree(d_dW);

        threads = 256;
        blocks = (outputSize + threads - 1) / threads;
        update_kernel<<<blocks, threads>>>(biases[layer], d_delta, learning_rate, outputSize);
        cudaDeviceSynchronize();

        // this will go to the next iteration
        float* d_prev;
        cudaMalloc(&d_prev, inputSize * sizeof(float));
        cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    inputSize, 1, outputSize,
                    &alpha,
                    weights[layer], outputSize,
                    d_delta, outputSize,
                    &beta,
                    d_prev, inputSize);
        
        threads = 256;
        blocks = (inputSize + threads - 1) / threads;

        if (layer > 0) {
            activation_deriv_kernel<<<blocks, threads>>>(activations[layer - 1], d_prev, inputSize);
            cudaDeviceSynchronize();
        }
        
        // honestly i dont know if we even need to free here
        if (layer != numLayers - 1) {
            cudaFree(d_delta);
        }
        d_delta = d_prev;
    }
    cudaFree(d_delta);
}


void MLP::log_prob(float* d_policy_output, float* d_actions, float* d_log_probs) {
    int n = output_dim;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    log_prob_kernel<<<blocks, threads>>>(d_policy_output, d_actions, d_log_probs, n);
    cudaDeviceSynchronize();
}

void MLP::ratio(float* d_new_log_probs, float* d_old_log_probs, float* d_ratios) {
    int n = output_dim;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    ratio_kernel<<<blocks, threads>>>(d_new_log_probs, d_old_log_probs, d_ratios, n);
    cudaDeviceSynchronize();
}

void MLP::surrogate_loss(float* d_ratios, float* d_advantages, float* d_surrogate) {
    int n = output_dim;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    float epsilon = 0.2f; // still hard coding this :skull: will update at some point
    surrogate_loss_kernel<<<blocks, threads>>>(d_ratios, d_advantages, d_surrogate, n, epsilon);
    cudaDeviceSynchronize();
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

void compute_value_grad(const float* d_predictions, const float* d_returns, float* d_grads, int n) {
    // this probably doesn't work as intended right now if im being honest
    // some issues with n value and empty space in memory i will fix in the future.

    // or actually it might work fine if I just update the rollout struct with a variable that holds the number of steps
    // and then pass that in here


    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    compute_value_grad_kernel<<<blocks, threads>>>(d_predictions, d_returns, d_grads, n);
    cudaDeviceSynchronize();
}
