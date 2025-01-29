/**
 * 
 * Environment implementation
 * 
 * very simple for now, vecenv does not have much support
 * 
 * 
 * 
 */


#include "env.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

// initializes random states
__global__ void init_curand_states(curandState* d_curand_states, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &d_curand_states[id]);
}

// resets the environment
__global__ void reset_kernel(float* d_state, int state_size, curandState* d_curand_states) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < state_size) {
        d_state[id] = curand_uniform(&d_curand_states[id]);
    }
}

// takes a step in the env
__global__ void step_kernel(float* action, float* d_state, float* d_next_state, float* d_reward, bool* d_done, int state_size, int action_size) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < state_size) {
        int action_index = id % action_size;
        d_next_state[id] = d_state[id] + 0.1f * action[action_index];
    }
    if (id == 0) {
        *d_reward = -1.0f; // i will replace this with an actual reward function at some point but i need to first make the envs work correctly
        *d_done = false;
    }
}

// constructor
Environment::Environment(int state_size, int action_size) : state_size(state_size), action_size(action_size) {
    cudaMalloc(&d_state, state_size * sizeof(float));
    cudaMalloc(&d_curand_states, state_size * sizeof(curandState));
    init_curand_states<<<1, state_size>>>(d_curand_states, time(0));
}

// destructor
Environment::~Environment() {
    cudaFree(d_state);
    cudaFree(d_curand_states);
}

void Environment::reset(float* d_state) {
    reset_kernel<<<1, state_size>>>(d_state, state_size, d_curand_states);
}

// Step function
void Environment::step(float* action, float* d_next_state, float* d_reward, bool* d_done) {
    step_kernel<<<1, state_size>>>(action, d_state, d_next_state, d_reward, d_done, state_size, action_size);
}





