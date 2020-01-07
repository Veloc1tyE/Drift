#pragma once

#include <vector>
#include <torch/torch.h>
#include "cpprl/model/nn.base.h"

using namespace torch;

namespace cpprl {
    
    
    class MlpBase : public NNBase {
        
        /*
         * An MLP for use in the A2C algorithm contains an actor, critic,
         * and requires a list of inputs
         */

        private:
            nn::Sequential actor;
            nn::Sequential critic;
            nn::Linear critic_linear;
            unsigned int num_inputs;

        public:
            
            MlpBase(unsigned int num_inputs,
                    bool recurrent = false;
                    unsigned int hidden_size = 64);

            std::vector<torch::Tensor> forward(torch::Tensor inputs,
                                               torch::Tensor hxs,
                                               torch::Tensor masks);
            
            inline unsigned int get_num_inputs() const { return num_inputs; }
    };
}
