/**
 * 
 * Environment implementation
 * 
 * very simple for now, vecenv does not have much support
 * 
 * This environement is a simple pendulum one, will make it more diverse in the future
 * 
 * 
 */


#include "env.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <iostream>

#define PI 3.1415926535f
#define MAX_TORQUE 2.0f
#define DT 0.05f // 20 Hz
#define G 9.81f
#define MASS 1.0f
#define LENGTH 1.0f

// initializes random states
__global__ void init_curand_states(curandState *states, unsigned long seed) {
    int id = threadIdx.x;
    curand_init(seed, id, 0, &states[id]);
}

// resets the environment
__global__ void reset_kernel(float* d_state, int state_size, curandState* d_curand_states) {
    int id = threadIdx.x;
    if (id < state_size) {
        curandState local_state = d_curand_states[id];
        d_state[0] = curand_uniform(&local_state) * 2 * PI - PI;
        d_state[1] = 0.0f;  // set omega to 0
        d_curand_states[id] = local_state;
    }
}

// takes a step in the env
__global__ void step_kernel(float* action, float* d_state, float* d_next_state, float* d_reward, bool* d_done, int state_size, int action_size) {
    int id = threadIdx.x;
    if (id == 0) {
        float theta = d_state[0];
        float omega = d_state[1];
        float torque = max(-MAX_TORQUE, min(MAX_TORQUE, action[0]));

        float alpha = (-3.0f * G / (2.0f * LENGTH)) * sinf(theta) + (3.0f / (MASS * LENGTH * LENGTH)) * torque;
        float new_omega = omega + alpha * DT;
        float new_theta = theta + new_omega * DT;
        
        // bound theta [-pi, pi]
        if (new_theta > PI) new_theta -= 2 * PI;
        if (new_theta < -PI) new_theta += 2 * PI;
        
        d_next_state[0] = new_theta;
        d_next_state[1] = new_omega;
        
        // Reward function: - (θ^2 + 0.1 * ω^2 + 0.001 * torque^2)
        float reward = -(theta * theta + 0.1f * omega * omega + 0.001f * torque * torque);
        
        d_reward[0] = reward;
        d_done[0] = false;

        d_state[0] = new_theta;
        d_state[1] = new_omega;
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
    current_step = 1;
}

// Step function
void Environment::step(float* action, float* d_next_state, float* d_reward, bool* d_done) {
    step_kernel<<<1, state_size>>>(action, d_state, d_next_state, d_reward, d_done, state_size, action_size);
    current_step++;
    
    if (current_step >= max_steps) {
        bool h_done = true;
        cudaMemcpy(d_done, &h_done, sizeof(bool), cudaMemcpyHostToDevice);
    }
}





