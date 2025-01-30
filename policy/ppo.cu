/**
 * ppo.cu
 * 
 * 
 */


#include "ppo.h"

PPO::PPO(Environment* env) {
    // setup a default network architecture
    std::vector<int> layers = {128, 256, 128, 64};

    this->policy_network = new MLP(layers);
    this->value_network = new MLP(layers);
    this->env = env;

}

PPO::PPO(Environment* env, std::vector<int> policy_network_dims, std::vector<int> value_network_dims) {

    this->policy_network = new MLP(policy_network_dims);
    this->value_network = new MLP(value_network_dims);
    this->env = env;

}


void PPO::train(int num_timesteps) {
    // TODO: update this function
    // the current implementation was just a sanity check

    float h_input[128];
    float h_output[128];
    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, 128 * sizeof(float));
    cudaMalloc(&d_output, 64 * sizeof(float));

    for (int i = 0; i < 128; i++) {
        h_input[i] = 1.0;
        h_output[i] = 0.0;
    }


    cudaMemcpy(d_input, &h_input, 128 * sizeof(float), cudaMemcpyHostToDevice);

    policy_network->forward(d_input, d_output);

    std::cout << "Printing output" << std::endl;

    cudaMemcpy(&h_output, d_output, 64 * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 64; i++) {
        std::cout << std::fixed << std::setprecision(4) << h_output[i] << " ";
    }


    cudaFree(d_input);
    cudaFree(d_output);
}

