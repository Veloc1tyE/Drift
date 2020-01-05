#pragma once

#include <memory>

#include <c10/util/ArrayRef.h>
#include <torch/torch.h>

#include <drift/generators/generator.h>
#include <drift/spaces.h>

namespace drift {
    
    /**
     * When the agent is initialised, we want it to be able to store all the information
     * it needs for training effectively, this class provides the infrastructure to do so
     */
    
    class RolloutStorage
    {
        // the A2C algorithm relies on tensors containing the following information
        private:
            // these represent the agent's knowledge of its environment
            torch::Tensor observations, hidden_states, rewards,
                value_predictions, returns, action_log_probs, actions, masks;
            // training related information
            torch::Device device; 
            int64_t num_steps;    
            int64_t step;
        
        public:       
            /*
             * Constructs for initialising this class
             */ 
            
            RolloutStorage(int64_t num_steps,
                        int64_t num_processes,
                        c10::ArrayRef<int64_t> obs_shape,
                        ActionSpace action_space,
                        int64_t hidden_state_size,
                        torch::Device device);
            
            // needed for storing multiple gamestates, as occurs in A2C 
            RolloutStorage(std::vector<RolloutStorage *> individual_storages, torch::Device device);
            
            /*
             * Define various methods required by RolloutStorage class,
             * more generally for use in RL algorithms
             */ 
            
            void after_update();
            void compute_returns(torch::Tensor next_value,
                                 bool use_gae,
                                 float gamma,
                                 float tau);
            
            // generate minibatches for training feed forward network
            std::unique_ptr<Generator> feed_forward_generator(torch::Tensor advantages,
                    int num_mini_batch);
            // generate minibatches for training recurren network
            std::unique_ptr<Generator> recurrent_generator(torch::Tensor advantages, int num_mini_batch);

            // store an observation for training
            void insert(torch::Tensor observation,
                        torch::Tensor hidden_state,
                        torch::Tensor action,
                        torch::Tensor action_log_prob,
                        torch::Tensor value_prediction,
                        torch::Tensor reward,
                        torch::Tensor mask);

            // initialise A2C training algorithm
            void set_first_observation(torch::Tensor observation);
            void to(torch::Device device);

            /*
             * Define constructs for retrieving all relevant information
             * for training in the minibatch
             */

            inline const torch::Tensor &get_actions() const { return actions; }
            inline const torch::Tensor &get_action_log_probs() const { return action_log_probs; }
            inline const torch::Tensor &get_hidden_states() const { return hidden_states; }
            inline const torch::Tensor &get_masks() const { return masks; }
            inline const torch::Tensor &get_observations() const { return observations; }
            inline const torch::Tensor &get_returns() const { return returns; }
            inline const torch::Tensor &get_rewards() const { return rewards; }
            inline const torch::Tensor &get_value_predictions() const { 
                return value_predictions; 
            }

            /*
             * Now set the mirrored methods for storing all relevant information in memory
             * to be used for training the algorithm
             */ 

            inline void set_actions(torch::Tensor actions) { this->actions = actions; }

            inline void set_action_log_probs(torch::Tensor action_log_probs)  {
                this->action_log_probs = action_log_probs; 
            }
            
            inline void set_hidden_states(torch::Tensor hidden_states) { 
                this->hidden_states = hidden_states; 
            }

            inline void set_masks(torch::Tensor masks) { this->masks = masks; }
            
            inline void set_observations(torch::Tensor observations) {
                this->observations = observations;
            }

            inline void set_rewards(torch::Tensor rewards) {
                this->rewards = rewards;
            }

            inline void set_value_predictions(torch::Tensor value_predictions) {
                this->value_predictions = value_predictions;
            }
            

    };
}


