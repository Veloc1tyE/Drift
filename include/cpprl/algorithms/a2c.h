#pragma once

#include <string>
#include <vector>

#include <torch/torch.h>

#include "cpprl/algorithms/algorithm.h"

namespace cpprl  {
    
    // Define the Policy used by the A2C algorithm given the gamestate
    class Policy;
    
    // Store all the information obtained during rollouts for use in Policy optimisation
    class RolloutStorage;
    
    /*
     * A2C uses MiniBatches, various agents, and synchronicity between these agents to train
     * the master network. It inherits all the information it needs for training from Algorithm.
     * Here we define all the utilities for training the A2C Algorithm given that.
     */ 

    class A2C : public Algorithm {
        private:
            // store reference to A2C policy
            Policy &policy;
            // variables for use in training policy
            float actor_loss_coef, value_loss_coef, entropy_coef, max_grad_norm, original_learning_rate;
            // point to learning algorithm used to train
            std::unique_ptr<torch::optim::RMSprop> optimizer;

        public:

            /*
             * Define the constructor for an A2C instance,
             * contains all information required for optimising the policy
             */ 

            A2C(Policy &policy,
                float actor_loss_coef,
                float value_loss_coef,
                float entropy_coef,
                float learning_rate,
                float epsilon = 1e-8,
                float alpha = 0.99,
                float max_grad_norm = 0.5);

            /*
             * We need to create a method to optimise A2C given its internal variables
             * as well as training data obtained from rollouts
             */ 

            std::vector<UpdateDatum> update(RolloutStorage &rollouts, float decay_level = 1);

    };
}
