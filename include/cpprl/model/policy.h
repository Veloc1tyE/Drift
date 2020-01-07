#pragma once

#include <vector>
#include <memory>

#include <torch/torch.h>

#include "drift/model/nn_base.h"
#include "drift/model/output_layers.h"
#include "drift/observation_normalizer.h"
#include "drift/spaces.h"

using namespace torch;

/*
 * Construct for Model Policy
 */ 

namespace drift {

    class PolicyImpl : public nn::Module {
        private:
            /* 
             * Policy requires an action_space, base for its neural-network model,
             * an observation_normalizer for observations flowing through this base,
             * and an output layer which defines the probability distribution over the action_space
             */

            // store the action space
            ActionSpace action_space;
            // reference to nn-base
            std::shared_ptr<NNBase> base;
            // normalise observations
            ObservationNormalizer observation_normalizer;
            // reference to nn-output-layer
            std::shared_ptr<OutputLayer> output_layer;
            
            // forward operation, assuming recurrency (take into account past information)
            std::vector<torch::Tensor> forward_gru(torch::Tensor x,
                                                   torch::Tensor y,
                                                   torch::Tensor masks);
        public:
            // Implement the policy constructor
            PolicyImpl(ActionSpace action_space,
                       std::shared_ptr<NNBase> base,
                       bool normalize_observations = false);
            
            // Take an action given the inputs, rnn hidden states and masks
            std::vector<torch::Tensor> act(torch::Tensor inputs,
                                           torch::Tensor rnn_hxs,
                                           torch::Tensor masks) const;
            
            // Evaluate the action taken based on the network inputs according to associated returns
            std::vector<torch::Tensor> evaluate_actions(torch::Tensor inputs,
                                                        torch::Tensor rnn_hxs,
                                                        torch::Tensor masks,
                                                        torch::Tensor actions) const;

            // retrieve the probability distribution over the action-space given network inputs
            torch::Tensor get_probs(torch::Tensor inputs,
                                    torch::Tensor rnn_hxs,
                                    torch::Tensor masks) const;

            // retrieve values produced from sampling probability distributions
            torch::Tensor get_values(torch::Tensor inputs,
                                     torch::Tensor rnn_hxs,
                                     torch::Tensor masks) const;

            // mean and variance of normaliser changes following each observation
            void update_observation_normalizer(torch::Tensor observation);
            
            inline bool is_recurrent() const { return base->is_recurrent(); }
            
            inline unsigned int get_hidden_size() const {
                return base->get_hidden_size();
            }
            
            inline bool using_observation_normalizer() const { 
                return !observation_normalizer.is_empty();
            }
    };

    // register policy to torch-module to enable easy tracking
    TORCH_MODULE(Policy);

}
