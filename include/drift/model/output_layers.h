#pragma once

#include <memory>
#include <torch/torch.h>
#include "drift/distributions/distribution.h"

using namespace torch;

namespace drift {
    
    /*
     * Actor/Critic networks map to an output layer over an action space
     * These will contain some probability distribution for the given space
     */ 

    class OutputLayer : public nn::Module {
        public:
            virtual ~OutputLayer() = 0;
            // Forward is just a reference to the output distribution
            virtual std::unique_ptr<Distribution> forward(torch::Tensor x) = 0;
    };

    inline OutputLayer::~OutputLayer() {};

    /*
     * Output may map to a Bernoulli distribution
     */ 

    class BernoulliOutput : public OutputLayer {
        private:
            nn::Linear linear;

        public:
            // We construct a Bernoulli mapping over the number of outputs given the input shape
            BernoulliOutput(unsigned int num_inputs, unsigned int num_outputs);
            
            // Reference output distribution flowing from the inputs
            std::unique_ptr<Distribution> forward(torch::Tensor x);
    };
    
    /*
     * Output may map to a Categorical distribution
     */ 

    class CategoricalOutput : public OutputLayer {
        private:
            nn::Linear linear;

        public:
            CategoricalOutput(unsigned int num_inputs, unsigned int num_outputs);
            std::unique_ptr<Distribution> forward(torch::Tensor x);
    };

    /*
     * We may have a normally distributed output distribution
     */ 

    class NormalOutput : public OutputLayer {
        private:
            // Normal distribution is defined by its mean and variance
            nn::Linear linear_loc;
            torch::Tensor scale_log;

        public:
            NormalOutput(unsigned int num_inputs, unsigned int num_outputs);

            std::unique_ptr<Distribution> forward(torch::Tensor x);

    };

    /*
     * Note: Further distributions are possible, like a normalising flow, to be implemented later
     */ 

}


