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

class MLP {
public:
    // MLP();
    MLP(const std::vector<int>& layers);
    ~MLP();

    void forward(float* input, float* output);

private:
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;
    std::vector<int> layerSizes;

    std::vector<float*> weights, biases;
    std::vector<float*> activations; // Stores intermediate outputs

    int numLayers;

    void initializeLayers();
    void cleanup();
};

#endif // NETWORK_H

