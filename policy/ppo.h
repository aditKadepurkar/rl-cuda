/**
 * 
 * PPO.h
 * 
 * 
 * 
 */


#ifndef PPO_H
#define PPO_H

#include "../env/env.h"
#include "network.h"
#include <cudnn.h>
#include <iostream>
#include <iomanip>

struct Rollout {
    float* states;
    float* actions;
    float* rewards;
    float* values;
    float* log_probs;
    float* advantages;
    float* returns;
};

class PPO {
public:
    __host__ PPO(Environment* env);
    __host__ PPO(Environment* env, std::vector<int> policy_network_dims, std::vector<int> value_network_dims);
    __host__ PPO(VecEnv env, std::vector<int> policy_network_dims, std::vector<int> value_network_dims);
    __host__ void train(int num_timesteps);
    __host__ void collect_rollouts();

private:
    MLP* policy_network;
    MLP* value_network;
    Environment* env;
    std::vector<Rollout> rollout_buffer;
    int ROLLOUT_BUFFER_SIZE = 100;
    int env_max_steps = 100;
    bool is_vecenv = false;
};


#endif // PPO_H
