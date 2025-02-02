/**
 * ppo.cu
 * 
 * 
 */


#include "ppo.h"

PPO::PPO(Environment* env) {
    // setup a default network architecture
    std::vector<int> layers = {env->state_size, 16, 8, env->action_size};

    this->policy_network = new MLP(layers);
    this->value_network = new MLP(layers);

    std::cout << "Value Network Output dim: " << value_network->output_dim << std::endl;

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

    int epoch = num_timesteps / env_max_steps;

    int policy_output_dim = policy_network->output_dim;

    // float pi_loss = 0.0f;

    std::cout << "Training PPO for " << epoch << " epochs" << std::endl;

    for (int i = 0; i < epoch; ++i) {
        

        // collect rollouts -- will put this in a loop later, testing for now
        collect_rollouts(); // collect_rollouts


        // std::cout << "Collected rollouts " << rollout_buffer.size() << std::endl;

        std::cout << "Epoch " << i+1 << std::endl;

        /* The followign code shows how to access the data in the rollout buffer
        
        
        

        int state_size = env->state_size;

        std::vector<float> example_state(state_size);
        cudaMemcpy(example_state.data(), rollout_buffer[0].states, state_size * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "Example state: ";
        for (const auto& val : example_state) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
        */


        // now we do some training!!!!

        /* Basically my notes sheet for the next steps:
        - I currently have the rollouts which have all the following:
            - states
            - actions
            - rewards
            - values
            - log_probs
            - advantages
            - returns

        Now I just need to do the calculations? I think?
        
        
        */

        float *d_states;
        float *d_output;
        float *d_new_log_probs;
        float *d_old_log_probs;

        Rollout& rollout = rollout_buffer[0];


        cudaMalloc(&d_old_log_probs, env_max_steps * sizeof(float));
        cudaMemset(d_old_log_probs, 0, env_max_steps * sizeof(float));

        cudaMalloc(&d_states, env_max_steps * env->state_size * sizeof(float));
        cudaMalloc(&d_output, policy_output_dim * sizeof(float));

        cudaMemcpy(&d_states, rollout.states, env_max_steps * env->state_size * sizeof(float), cudaMemcpyDeviceToDevice);

        policy_network->forward(d_states, d_output);

        // check if d_output or rollout.states is NaN
        
        // float h_output[policy_output_dim];
        // cudaMemcpy(h_output, d_output, policy_output_dim * sizeof(float), cudaMemcpyDeviceToHost);

        // for (int j = 0; j < policy_output_dim; ++j) {
        //     if (std::isnan(h_output[j])) {
        //         std::cout << "Output is NaN" << std::endl;
        //         exit(1);
        //     }
        // }


        policy_network->log_prob(d_output, rollout.actions, d_old_log_probs);

        cudaMalloc(&d_new_log_probs, sizeof(float) * policy_output_dim);

        // int state_size = env->state_size;
        

        for (int j = 0; j < ROLLOUT_BUFFER_SIZE; ++j) {
            // std::cout << "Training on rollout " << j << std::endl;

            rollout = rollout_buffer[j];

            // get new log probs
            policy_network->forward(rollout.states, d_output);
            policy_network->log_prob(d_output, rollout.actions, d_new_log_probs);

            // check if new and old log probs are NaN
            // float h_old_log_probs[env_max_steps];
            // float h_new_log_probs[env_max_steps];
            // cudaMemcpy(h_old_log_probs, d_old_log_probs, env_max_steps * sizeof(float), cudaMemcpyDeviceToHost);
            // cudaMemcpy(h_new_log_probs, d_new_log_probs, env_max_steps * sizeof(float), cudaMemcpyDeviceToHost);

            // for (int k = 0; k < env_max_steps; ++k) {
            //     if (std::isnan(h_old_log_probs[k])) {
            //         std::cout << "Old log probs are NaN" << std::endl;
            //         exit(1);
            //     }
            //     // if (std::isnan(h_new_log_probs[i])) {
            //     //     std::cout << "New log probs are NaN" << std::endl;
            //     //     exit(1);
            //     // }
            // }

            // r_t(theta) = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)
            float *d_ratios;
            cudaMalloc(&d_ratios, policy_output_dim * sizeof(float));
            policy_network->ratio(d_new_log_probs, d_old_log_probs, d_ratios);


/*
            // check if any ratios or advnatages are NaN
            float h_ratios[policy_output_dim];
            cudaMemcpy(h_ratios, d_ratios, policy_output_dim * sizeof(float), cudaMemcpyDeviceToHost);

            for (int i = 0; i < policy_output_dim; ++i) {
                if (std::isnan(h_ratios[i])) {
                    std::cout << "Ratios are NaN" << std::endl;
                    exit(1);
                }
            }

            // advantages
            float h_advantages[env_max_steps];
            cudaMemcpy(h_advantages, rollout.advantages, env_max_steps * sizeof(float), cudaMemcpyDeviceToHost);
            
            for (int i = 0; i < env_max_steps; ++i) {
                if (std::isnan(h_advantages[i])) {
                    std::cout << "Advantages are NaN" << std::endl;
                    exit(1);
                }
            }
*/

            float *d_surrogate;
            cudaMalloc(&d_surrogate, policy_output_dim * sizeof(float));
            policy_network->surrogate_loss(d_ratios, rollout.advantages, d_surrogate);


            float h_surrogate[policy_output_dim];
            cudaMemcpy(h_surrogate, d_surrogate, policy_output_dim * sizeof(float), cudaMemcpyDeviceToHost);

            for (int i = 0; i < policy_output_dim; ++i) {
                if (std::isnan(h_surrogate[i])) {
                    std::cout << "Surrogate loss is NaN" << std::endl;
                    exit(1);
                }
            }

            // update the parameters 
            policy_network->backward(d_surrogate, policy_network->learning_rate);

            float *d_value_grads;
            cudaMalloc(&d_value_grads, policy_output_dim * sizeof(float));

            // caclulating the value grad with teh new func i made which I hope works correctly
            compute_value_grad(rollout.values, rollout.returns, d_value_grads, env_max_steps);

            value_network->backward(d_value_grads, value_network->learning_rate);


            // sum up the d_surrogate
            // float h_surrogate[policy_output_dim];
            cudaMemcpy(h_surrogate, d_surrogate, policy_output_dim * sizeof(float), cudaMemcpyDeviceToHost);


            FILE *f = fopen("pi_loss.txt", "a");

            // float max_val = h_surrogate[0];
            for (int k = 0; k < policy_output_dim; ++k) {
                fprintf(f, "%f ", h_surrogate[k]);
            }
            fprintf(f, "\n");
            
            


        /*


            // print d_value_grads and d_surrogate
            std::vector<float> h_value_grads(env_max_steps);
            std::vector<float> h_surrogate(env_max_steps);

            cudaMemcpy(h_value_grads.data(), d_value_grads, env_max_steps * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_surrogate.data(), d_surrogate, env_max_steps * sizeof(float), cudaMemcpyDeviceToHost);

            std::cout << "Value grads: ";
            for (const auto& val : h_value_grads) {
                std::cout << val << " ";
            }
            std::cout << std::endl;

            std::cout << "Surrogate: ";
            for (const auto& val : h_surrogate) {
                std::cout << val << " ";
            }
            std::cout << std::endl;

        */

            // Free allocated memory
            cudaFree(d_ratios);
            cudaFree(d_surrogate);
        }


    }

}

// #define CUDA_CHECK(call) \
//     do { \
//         cudaError_t err = call; \
//         if (err != cudaSuccess) { \
//             fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
//                     __FILE__, __LINE__, cudaGetErrorString(err)); \
//             exit(EXIT_FAILURE); \
//         } \
//     } while (0)

void PPO::collect_rollouts() {
    int action_dim = env->action_size;
    int state_size = env->state_size;

    for (int i = 0; i < ROLLOUT_BUFFER_SIZE; ++i) {
        Rollout& rollout = rollout_buffer[i];

        if (rollout.states == nullptr) {
            cudaMalloc(&rollout.states, env_max_steps * state_size * sizeof(float));
            cudaMalloc(&rollout.actions, env_max_steps * action_dim * sizeof(float));
            cudaMalloc(&rollout.rewards, env_max_steps * sizeof(float));
            cudaMalloc(&rollout.values, env_max_steps * sizeof(float));
            // CUDA_CHECK(cudaMalloc(&rollout.log_probs, env_max_steps * sizeof(float)));
            cudaMalloc(&rollout.advantages, env_max_steps * sizeof(float));
            cudaMalloc(&rollout.returns, env_max_steps * sizeof(float));
        }

        float* d_state;
        float* d_next_state;
        float* d_reward;
        bool* d_done;
        float* d_action;
        float* d_value;

        bool h_done;

        cudaMalloc(&d_state, state_size * sizeof(float));
        cudaMalloc(&d_next_state, state_size * sizeof(float));
        cudaMalloc(&d_reward, sizeof(float));
        cudaMalloc(&d_done, sizeof(bool));
        cudaMalloc(&d_action, action_dim * sizeof(float));
        cudaMalloc(&d_value, sizeof(float));

        int step = 0;
        h_done = false;
        env->reset(d_state);

        // std::cout << "Rollout " << i << std::endl;

        while (!h_done && step < env_max_steps) {
            // Generate action directly on device for now
            cudaMemset(d_action, 0.1, action_dim * sizeof(float));

            env->step(d_action, d_next_state, d_reward, d_done);

            value_network->forward(d_state, d_value);

            // Store rollout data on the device
            cudaMemcpy(&rollout.states[step * state_size], d_state, state_size * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(&rollout.actions[step * action_dim], d_action, action_dim * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(&rollout.rewards[step], d_reward, sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(&rollout.values[step], d_value, sizeof(float), cudaMemcpyDeviceToDevice);

            cudaMemcpy(&h_done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);

            cudaMemcpy(d_state, d_next_state, state_size * sizeof(float), cudaMemcpyDeviceToDevice);
            step++;
        }

        // std::cout << "Step " << step << std::endl;

        // Compute advantages, values, etc. on the device
        for (int k = 0; k < step; k++) {
            float reward, value, next_value;

            cudaMemcpy(&reward, &rollout.rewards[k], sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&value, &rollout.values[k], sizeof(float), cudaMemcpyDeviceToHost);
            // std::cout << "Value: " << value << std::endl;

            // next_value = (k == step - 1) ? 0 : rollout.values[k + 1];
            
            if (k == step - 1) {
                next_value = 0;
            } else {
                cudaMemcpy(&next_value, &rollout.values[k + 1], sizeof(float), cudaMemcpyDeviceToHost);
            }


            // these are hyperparameters that I should probably make into variables
            // but for now they are hardcoded
            float delta = reward + 0.99 * next_value - value;
            


            // lotss of optimization that can be done here very easily by not wasting all the time copying back and forth
            // TODO later

            float advantage;
            if (k == 0) {
                advantage = delta;
            } else {
                float prev_advantage;
                cudaMemcpy(&prev_advantage, &rollout.advantages[k - 1], sizeof(float), cudaMemcpyDeviceToHost);
                advantage = delta + 0.99 * 0.95 * prev_advantage;
            }
            cudaMemcpy(&rollout.advantages[k], &advantage, sizeof(float), cudaMemcpyHostToDevice);

            float ret;
            if (k == 0) {
                ret = reward;
            } else {
                float prev_return;
                cudaMemcpy(&prev_return, &rollout.returns[k - 1], sizeof(float), cudaMemcpyDeviceToHost);
                ret = reward + 0.99 * prev_return;
            }
            cudaMemcpy(&rollout.returns[k], &ret, sizeof(float), cudaMemcpyHostToDevice);
        }


        step = 0;

        // Free device memory for this rollout
        cudaFree(d_state);
        cudaFree(d_next_state);
        cudaFree(d_reward);
        cudaFree(d_done);
        cudaFree(d_action);
    }
}