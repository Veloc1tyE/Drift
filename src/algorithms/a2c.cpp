#include <chrono>
#include <memory>

#include <torch/torch.h>

#include "drift/algorithms/a2c.h"
#include "drift/algorithms/algorithm.h"
#include "drift/model/policy.h"
#include "drift/storage.h"
#include "drift/spaces.h"


namespace drift {
    A2C::A2C(Policy &policy,
             float actor_loss_coef,
             float value_loss_coef,
             float entropy_coef,
             float learning_rate,
             float epsilon,
             float alpha,
             float max_grad_norm)
          // initialise parameters as defined in a2c.h
        
        : policy(policy), // action-space distribution given input
          // loss parameters
          actor_loss_coef(actor_loss_coef), 
          value_loss_coef(value_loss_coef),
          // regularisation parameters
          entropy_coef(entropy_coef),
          max_grad_norm(max_grad_norm),
          original_learning_rate(learning_rate),
          // use RMSprop as the optimisation algorithm for stability
          optimizer(std::make_unique<torch::optim::RMSprop>(
                
                // Reference policy as parameters since this is our optimisation target      
                policy->parameters(),
                torch::optim::RMSpropOptions(learning_rate) // step size for RMSprop, 
                            // define stable optimisation based on RMS propagation of steps
                            // with hyperparameters
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
        obs_shape.insert(obs_shape.begin(), -1);
        
    }

}
