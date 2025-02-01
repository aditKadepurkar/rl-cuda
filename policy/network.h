/**
 * network.h
 * 
 * 
 */


#ifndef NETWORK_H
#define NETWORK_H

#include <cudnn.h>
#include <cublas_v2.h>
#include <vector>


void compute_value_grad(const float* d_values, const float* d_returns, float* d_value_grad, int n);

class MLP {
public:
    // MLP();
    MLP(const std::vector<int>& layers);
    ~MLP();

    void forward(float* input, float* output);
    void backward(float* d_output_grad, float learning_rate);
    void log_prob(float* d_policy_output, float* d_actions, float* d_log_probs);
    void ratio(float* d_new_log_probs, float* d_old_log_probs, float* d_ratios);
    void surrogate_loss(float* d_ratios, float* d_advantages, float* d_surrogate);


    int output_dim;
    float learning_rate = 1e-3;
    float* d_input;

private:
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;
    std::vector<int> layerSizes;

    std::vector<float*> weights, biases;
    std::vector<float*> activations; // intermediate activations(don't actually have this owrking rn)

    int numLayers;

    void initializeLayers();
    void cleanup();
};

#endif // NETWORK_H

