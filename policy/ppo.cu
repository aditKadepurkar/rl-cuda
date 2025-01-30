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
    this->env_max_steps = env->max_steps;
    this->rollout_buffer = std::vector<Rollout>(ROLLOUT_BUFFER_SIZE);

}

PPO::PPO(Environment* env, std::vector<int> policy_network_dims, std::vector<int> value_network_dims) {

    this->policy_network = new MLP(policy_network_dims);
    this->value_network = new MLP(value_network_dims);
    this->env = env;
    this->env_max_steps = env->max_steps;
    this->rollout_buffer = std::vector<Rollout>(ROLLOUT_BUFFER_SIZE);
}


void PPO::train(int num_timesteps) {

    std::cout << "Training PPO for " << num_timesteps << " timesteps" << std::endl;

    // collect rollouts -- will put this in a loop later, testing for now
    collect_rollouts();


    std::cout << "Collected rollouts " << rollout_buffer.size() << std::endl;

    // std::cout << rollout_buffer[0].states[0] << std::endl;

    // now we do some training!!!!


}

void PPO::collect_rollouts() {
    // this should basically be the actor process

    int action_dim = env->action_size;
    int state_size = env->state_size;

    // allocate rollout on the host
    Rollout* rollout = new Rollout();
    rollout->states = (float*)malloc(env_max_steps * state_size * sizeof(float));
    rollout->actions = (float*)malloc(env_max_steps * action_dim * sizeof(float));
    rollout->rewards = (float*)malloc(env_max_steps * sizeof(float));
    rollout->values = (float*)malloc(env_max_steps * sizeof(float));
    rollout->log_probs = (float*)malloc(env_max_steps * sizeof(float));
    rollout->advantages = (float*)malloc(env_max_steps * sizeof(float));
    rollout->returns = (float*)malloc(env_max_steps * sizeof(float));

    float* d_state;
    float* d_next_state;
    float* d_reward;
    bool* d_done;
    float* d_action;

    bool h_done;

    cudaMalloc(&d_state, state_size * sizeof(float));
    cudaMalloc(&d_next_state, state_size * sizeof(float));
    cudaMalloc(&d_reward, sizeof(float));
    cudaMalloc(&d_done, sizeof(bool));
    cudaMalloc(&d_action, action_dim * sizeof(float));

    int step = 0;

    for (int i = 0; i < ROLLOUT_BUFFER_SIZE; i++) {
        std::cout << "Rollout " << i << std::endl;
        h_done = false;
        env->reset(d_state);
    
        while (!h_done) {

            // generate action directly on device for now
            cudaMemset(d_action, 0.1, action_dim * sizeof(float));

            env->step(d_action, d_next_state, d_reward, d_done);

            cudaMemcpy(&h_done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);

            //  adding the rollout data
            cudaMemcpy(&rollout->states[step * state_size], d_state, state_size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&rollout->actions[step * action_dim], d_action, action_dim * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&rollout->rewards[step], d_reward, sizeof(float), cudaMemcpyDeviceToHost);

            cudaMemcpy(d_state, d_next_state, state_size * sizeof(float), cudaMemcpyDeviceToDevice);
            step++;
        }
        
        std::cout << "Rollout complete" << std::endl;

        // add rollout data to the buffer
        rollout_buffer[i] = *rollout;

        step = 0;
    }

    // free device memory
    cudaFree(d_state);
    cudaFree(d_next_state);
    cudaFree(d_reward);
    cudaFree(d_done);
    cudaFree(d_action);

    // free hsot memory
    // free(rollout.states);
    // free(rollout.actions);
    // free(rollout.rewards);
    // free(rollout.values);
    // free(rollout.log_probs);
    // free(rollout.advantages);
    // free(rollout.returns);
}
