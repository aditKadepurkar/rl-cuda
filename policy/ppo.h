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

class PPO {
public:
    __host__ PPO(Environment* env);
    __host__ PPO(Environment* env, std::vector<int> policy_network_dims, std::vector<int> value_network_dims);
    __host__ PPO(VecEnv env, std::vector<int> policy_network_dims, std::vector<int> value_network_dims);

    __host__ void train(int num_timesteps);


private:
    MLP* policy_network;
    MLP* value_network;
    Environment* env;
};





#endif // PPO_H
