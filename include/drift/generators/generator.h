#pragma once

#include <vector>

#include <torch/torch.h>

namespace drift
{
    /*
     * Define structure used for mini-batch processing and training
     * Training should proceed through MiniBatches for stability.
     */ 
    struct MiniBatch
    {
        // include all variables required by A2C algorithm for training
        torch::Tensor observations, hidden_states, actions, value_predictions,
            returns, masks, action_log_probs, advantages;
        
        MiniBatch() {}
        MiniBatch(torch::Tensor observations,
                torch::Tensor hidden_states,
                torch::Tensor actions,
                torch::Tensor value_predictions,
                torch::Tensor returns,
                torch::Tensor masks,
                torch::Tensor action_log_probs,
                torch::Tensor advantages)
            : observations(observations),
            hidden_states(hidden_states),
            actions(actions),
            value_predictions(value_predictions),
            returns(returns),
            masks(masks),
            action_log_probs(action_log_probs),
            advantages(advantages) {}
    };
    
    /*
     * Abstraction used to point to minibatches during training
     */

    class Generator
    {
        public:
            virtual ~Generator() = 0;

            virtual bool done() const = 0;
            virtual MiniBatch next() = 0;
    };

    inline Generator::~Generator() {}
}
