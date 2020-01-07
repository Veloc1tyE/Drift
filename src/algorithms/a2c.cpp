#include <chrono>
#include <memory>

#include <torch/torch.h>

#include "cpprl/algorithms/a2c.h"
#include "cpprl/algorithms/algorithm.h"
#include "cpprl/model/policy.h"
#include "cpprl/storage.h"
#include "cpprl/spaces.h"


namespace cpprl {
    A2C::A2C(Policy &policy,
             float actor_loss_coef,
             float value_loss_coef,
             float entropy_coef,
             float learning_rate,
             float epsilon,
             float alpha,
             float max_grad_norm)

        : policy(policy), // intialise policy
          
          // loss parameters
          actor_loss_coef(actor_loss_coef), 
          value_loss_coef(value_loss_coef),
          
          // regularisation parameters
          entropy_coef(entropy_coef),
          max_grad_norm(max_grad_norm),
          original_learning_rate(learning_rate),

          // use RMSprop as the optimisation algorithm for stability
          optimizer(std::make_unique<torch::optim::RMSprop>(      
                // This is our optimisation target      
                policy->parameters(),
                torch::optim::RMSpropOptions(learning_rate) 
                            // gradient propagation hyperparameters
                            .eps(epsilon) 
                            .alpha(alpha))) {}
    
    // Given rollout data, the optimiser and the policy, we need to optimise the policy based on the data
    std::vector<UpdateDatum> A2C::update(RolloutStorage &rollouts, float decay_level)  {
        // Decay learning-rate over time for stability
        optimizer->options.learning_rate(original_learning_rate * decay_level);

        /* 
        * Prepare observations for use in training
        */ 
        
        // retrieve the observations from the RolloutStorage class
        auto full_obs_shape = rollouts.get_observations().sizes(); // where sizes encodes dimension of observations

        std::vector<int64_t> obs_shape(full_obs_shape.begin() + 2, 
                                    full_obs_shape.end());
        
        // restructure observation shape
        obs_shape.insert(obs_shape.begin(), -1);
        
        // action_space consists of a type and shape, -1 is the indice of the shape
        auto action_shape = rollouts.get_actions().size(-1); 
        // rewards is a list of tensors, so sizes retrieves the list of dimensions
        auto rewards_shape = rollouts.get_rewards().sizes(); 
        // reward associated with every timestep, size of list of tensors => rewards shape
        int num_steps = rewards_shape[0];         
        // multiple agents running synchronously generate a list of rewards at each timestep
        int num_processes = rewards_shape[1];        
        
        /*
        * Update observation normaliser based on observations, 
        * since normalisation is based on the mean and variances of the samples
        */
        if (policy->using_observation_normalizer())  {
            policy->update_observation_normalizer(rollouts.get_observations());
        }

        // Run evaluation on rollouts
        auto evaluate_result = policy->evaluate_actions(
                // view observations as a list of tensors from start to finish
                rollouts.get_observations().slice(0, 0, -1).view(obs_shape),  
                
                // flatten hidden states into a list of tensors, encoding hidden state at each timestep    
                rollouts.get_hidden_states()[0].view({-1, policy->get_hidden_size()}),                
                // retrieve masks as a list of tensors from start to finish, as a flattened vector
                rollouts.get_masks().slice(0, 0, -1).view({-1, 1}),
                
                // get actions as a flattened list of action info
                rollouts.get_actions().view({-1, action_shape}));

        // retrieve the list of evauluations as a matrix of steps and processes, critic subset
        auto values = evaluate_result[0].view({num_steps, num_processes, 1});
        
        // retrieve the action distribution according to the actor
        auto action_log_probs = evaluate_result[1].view(
                {num_steps, num_processes, 1});

        // Calculate advantages
        auto advantages = rollouts.get_returns().slice(0, 0, -1) - values;

        // Value loss
        auto value_loss = advantages.pow(2).mean();

        // Action loss
        auto action_loss = -(advantages.detach() * action_log_probs).mean();

        // Total loss
        auto loss = (value_loss * value_loss_coef + action_loss 
                    - evaluate_result[2] * entropy_coef);

        // Step optimizer
        optimizer->zero_grad();
        loss.backward();
        optimizer->step();

        return {{"Value loss", value_loss.item().toFloat()},
                {"Action loss", action_loss.item().toFloat()},
                {"Entropy", evaluate_result[2].item().toFloat()}};

    }

    /*
    * Test cases to ensure algorithm is set up properly
    */ 

    static void learn_pattern(Policy &policy, RolloutStorage &storage, A2C &a2c) {
        /*
        * Here we train the agent to maximise its outputs,
        * given random integer input in [0,1]
        */

        // 10 episodes
        for (int i = 0; i < 10; i++)  {
            // 5 concurrent agents
            for (int j = 0; j < 5; j++) {
                auto observation = torch::randint(0, 2, {2, 1});
                std::vector<torch::Tensor> act_result;  
                {
                    torch::NoGradGuard no_grad;
                    act_result = policy->act(observation,
                                             torch::Tensor(),
                                             torch::ones({2,1}));
                }
                auto actions = act_result[1]; // action taken
                auto rewards = actions;
                storage.insert(observation,
                                // two concurrent processes and five steps
                                torch::zeros({2,5}),
                                actions,
                                act_result[2], // action log_probs
                                act_result[0], // value prediction
                                rewards,
                                // masks
                                torch::ones({2, 1}));
            }

            torch::Tensor next_value;
            {
                torch::NoGradGuard no_grad;
                // initialise next_value as a flattened list of observations, hidden states and masks
                next_value = policy->get_values(
                                        storage.get_observations()[-1],
                                        storage.get_hidden_states()[-1],
                                        storage.get_masks()[-1]).detach();
            } // generate policy based on the pattern in the loop
            
            // compute returns then update the network to learn the pattern
            storage.compute_returns(next_value, false, 0., 0.9);
            a2c.update(storage);
            storage.after_update();
        }
    }

    /*
    * Example game to ensure the agent is set up correctly
    */ 
    static void learn_game(Policy &policy, RolloutStorage &storage, A2C &a2c)  {
        /*
        * Test case: If the action matches the input, give a reward of 1, else -1
        */
        auto observation = torch::randint(0, 2, {2, 1});
        storage.set_first_observation(observation);
        
        for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < 5; ++j) {
                std::vector<torch::Tensor> act_result; 
                {
                    torch::NoGradGuard no_grad;
                    act_result = policy->act(observation,
                                                torch::Tensor(),
                                                torch::ones({2, 1}));
                }
                // retrieve the action, which will just be sampled from a 2D action space
                auto actions = act_result[1];
                // matching action to observation
                auto rewards = ((actions == observation.to(torch::kLong)).to(torch::kFloat) * 2) - 1;
                // random integer in [0,1]
                observation = torch::randint(0, 2, {2, 1});
                
                // store info relevant for training
                storage.insert(observation,
                                torch::zeros({2,5}), 
                                actions, 
                                act_result[2], 
                                act_result[0], 
                                rewards,
                                torch::ones({2,1}));
            }

            torch::Tensor next_value;
            {
                torch::NoGradGuard no_grad;
                next_value = policy->get_values(
                                        storage.get_observations()[-1],
                                        storage.get_hidden_states()[-1],
                                        storage.get_masks()[-1]).detach();          
            }
            storage.compute_returns(next_value, false, 0.1, 0.9);

            a2c.update(storage);
            storage.after_update();

        }

    }

    // Torch test processes
    
    TEST_CASE("A2C")
    {
        SUBCASE("update() learns basic pattern")
        {
            torch::manual_seed(0);
            auto base = std::make_shared<MlpBase>(1, false, 5);
            
            ActionSpace space{"Discrete", {2}}; // Algo must learn to produce second action, 
                                                // this is a discrete action space: [0,1]
            Policy policy(space, base);

            // 5 steps and 2 concurrent processes, these generate 1 tensor, running on CPU
            RolloutStorage storage(5, 2, {1}, space, 5, torch::kCPU);

            // intialise A2C algorithm
            A2C a2c(policy, 1, 0.5, 1e-3, 0.001);

            // the reward is the action
            auto pre_game_probs = policy->get_probs( // probability distribution of policy
                        torch::ones({2,1}),  // inputs
                        torch::zeros({2,5}), // no rnn hidden-state for 2 processes and 5 steps
                        torch::ones({2,5})); // no masking

            learn_pattern(policy, storage, a2c);
            
            auto post_game_probs = policy->get_probs(
                    torch::ones({2, 1}),
                    torch::zeros({2, 5}),
                    torch::ones({2, 1}));

            // print outputs and run checks with torch macros
            INFO("Pre-training probabilities: \n"
                    << pre_game_probs << "\n");
            INFO("Post-training probabilities: \n"
                    << post_game_probs << "\n");
            CHECK(post_game_probs[0][0].item().toDouble() <
                    pre_game_probs[0][0].item().toDouble()); // minimise probability of no reward
            CHECK(post_game_probs[0][1].item().toDouble() >
                    pre_game_probs[0][1].item().toDouble()); // maximise probability of reward
        }

        SUBCASE("update() learns basic game") 
        {
            SUBCASE("Without normalised observations") 
            {
                /*
                 * Here we're training the agent to reproduce the input
                 */ 
                torch::manual_seed(0);
                auto base = std::make_shared<MlpBase>(1, false, 5);
                ActionSpace space{"Discrete", {2}};

                Policy policy(space, base);
                RolloutStorage storage(5, 2, {1}, space, 5, torch::kCPU);
                A2C a2c(policy, storage, a2c);

                auto pre_game_probs = policy->get_probs(
                    torch::ones({2, 1}),
                    torch::zeros({2, 5}),
                    torch::ones({2, 1}));

                learn_game(policy, storage, a2c);

                auto post_game_probs = policy->get_probs(
                        torch::ones({2, 1}),
                        torch::zeros({2, 5}),
                        torch::ones({2, 1}));

                INFO("Pre-training probabilities: \n"
                        << pre_game_probs << "\n");
                INFO("Post-training probabilities: \n"
                        << post_game_probs << "\n");

                // Agent wants to reproduce input in second part of action-space
                // We include the first part of the action-space for easy testing
                CHECK(post_game_probs[0][0].item().toDouble() <
                        pre_game_probs[0][0].item().toDouble());
                
                // maximise the probability of the input
                CHECK(post_game_probs[0][1].item().toDouble() > 
                        pre_game_probs[0][1].item().toDouble());

            }
            SUBCASE("With normalized observations") 
            {
                torch::manual_seed(0);
                auto base = std::make_shared<MlpBase>(1, false, 5);
                ActionSpace space{"Discrete", {2}}; // observation is discrete, so is action-space
                
                Policy policy(space, base, true); // normalise observations for policy
                RolloutStorage storage(5, 2, {1}, space, 5, torch::kCPU);
                A2C a2c(policy, 1, 0.5, 1e-7, 0.0001);

                auto pre_game_probs = policy->get_probs(
                            torch::ones({2, 1}),
                            torch::zeros({2, 5}),
                            torch::ones({2, 1}));
                
                learn_game(policy, storage, a2c);

                auto post_game_probs = policy->get_probs(
                        torch::ones({2, 1}),
                        torch::zeros({2, 5}),
                        torch::ones({2, 1}));

                INFO("Pre-training probabilities: \n"
                        << pre_game_probs << "\n");
                INFO("Post-training probabilities: \n"
                        << post_game_probs << "\n");
                CHECK(post_game_probs[0][0].item().toDouble() <
                        pre_game_probs[0][0].item().toDouble());
                CHECK(post_game_probs[0][1].item().toDouble() >
                        pre_game_probs[0][1].item().toDouble());

            }
        }

    }

}
