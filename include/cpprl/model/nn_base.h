#pragma once

#include <vector>
#include <torch/torch.h>

using namespace torch;

namespace cpprl  {
    
    /*
     * Abstract class used to contruct Neural Networks
     */ 

    class NNBase : public nn::Module {
        private:
            // generalise recurrent unit, required for models based on recurrency
            nn::GRU gru;

            // attributes of NN
            unsigned int hidden_size;
            bool recurrent;

        public:
            /* required methods
             */

            // constructor
            NNBase(bool recurrent,
                   unsigned int recurrent_input_size,
                   unsigned int hidden_size);
            
            // mapping input to output, here virtual is used as this is generally applicable
            virtual std::vector<torch::Tensor> forward(torch::Tensor inputs,
                                                       torch::Tensor hxs,
                                                       torch::Tensor masks);
            // similar mapping but with recurrency
            std::vector<torch::Tensor> forward_gru(torch::Tensor inputs,
                                                   torch::Tensor hxs,
                                                   torch::Tensor masks);
            // retrieve info about NN
            unsigned int get_hidden_size() const;
            inline int get_output_size() const { return hidden_size; }
            inline bool is_recurrent() const { return recurrent; }
    };

}
