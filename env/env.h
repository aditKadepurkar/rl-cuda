

#ifndef ENV_H
#define ENV_H

#include <cuda_runtime.h>
#include <vector>
#include <curand_kernel.h>

class Environment {
public:
    __host__ Environment(int state_size, int action_size);
    __host__ ~Environment();

    __host__ void reset(float* d_state);
    __host__ void step(float* action, float* d_next_state, float* d_reward, bool* d_done);

private:
    int state_size;
    int action_size;
    float* d_state;
    curandState* d_curand_states;
};

// CUDA kernel functions
__global__ void reset_kernel(float* d_state, int state_size, curandState* d_curand_states);
__global__ void step_kernel(float* action, float* d_state, float* d_next_state, float* d_reward, bool* d_done, int state_size, int action_size);
__global__ void init_curand_states(curandState* d_curand_states, unsigned long seed);

#endif // ENV_H

