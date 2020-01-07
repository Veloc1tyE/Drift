#pragma once

#include <torch/torch.h>

namespace drift {
    
    /*
     * Distribution abstraction for use in distributing the policy of A2C
     * Needed as Actor-Critic has a probabilistic policy
     */ 

    class Distribution {
        // distribution for policy occurs over a batch of events and has a distribution shape
        protected:
            std::vector<int64_t> batch_shape, event_shape;
            std::vector<int64_t> extended_shape(c10::ArrayRef<int64_t> sample_shape);

        public:
            // abstract attributes of a given probability distribution
            virtual ~Distribution() = 0;
            virtual torch::Tensor entropy() = 0;
            virtual torch::Tensor log_prob(torch::Tensor value) = 0;
            virtual torch::Tensor sample(c10::ArrayRef<int64_t> sample_shape = {}) = 0;
    };

    inline Distribution::~Distribution() {}
}
