#include "env/env.h"
#include "policy/ppo.h"
#include <iostream>
#include <cstdlib>
#include <iomanip>

#define NUM_STEPS 100

int main() {
    int state_size = 2;
    int action_dim = 1;

    Environment env(state_size, action_dim);

    float* d_state;
    float* d_next_state;
    float* d_reward;
    bool* d_done;
    float* d_action;

    cudaMalloc(&d_state, state_size * sizeof(float));
    cudaMalloc(&d_next_state, state_size * sizeof(float));
    cudaMalloc(&d_reward, sizeof(float));
    cudaMalloc(&d_done, sizeof(bool));
    cudaMalloc(&d_action, action_dim * sizeof(float));

    PPO ppo(&env);

    env.reset(d_state);

    std::cout << "Environment reset complete." << std::endl;


    std::cout << "Training has started." << std::endl;
    ppo.train(10000);
    std::cout << "Training has finished." << std::endl;




    /* Code to see how the env runs.

    float h_state[state_size];
    float h_action[action_dim];

    for (int i = 0; i < NUM_STEPS; i++) {

        for (int j = 0; j < action_dim; j++) {
            h_action[j] = static_cast<float>(rand()) / RAND_MAX;
        }
        cudaMemcpy(d_action, h_action, action_dim * sizeof(float), cudaMemcpyHostToDevice);


        env.step(d_action, d_next_state, d_reward, d_done);

        cudaMemcpy(h_state, d_state, state_size * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "Step " << i + 1 << " | Action: [ ";
        for (int j = 0; j < action_dim; j++) {
            std::cout << std::fixed << std::setprecision(4) << h_action[j] << " ";
        }
        std::cout << "] | State: [ ";
        for (int j = 0; j < state_size; j++) {
            std::cout << std::fixed << std::setprecision(4) << h_state[j] << " ";
        }
        std::cout << "]" << std::endl;

        bool h_done;
        cudaMemcpy(&h_done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);

        if (h_done) {
            std::cout << "Episode finished early at step " << i + 1 << std::endl;
            break;
        }

        cudaMemcpy(d_state, d_next_state, state_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    std::cout << "Simulation complete." << std::endl;

    */

    cudaFree(d_state);
    cudaFree(d_next_state);
    cudaFree(d_reward);
    cudaFree(d_done);
    cudaFree(d_action);

    return 0;
}
