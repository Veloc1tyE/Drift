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
                value_predictions, action_log_probs, masks;
            // training related information
            torch::Device device; 
            int64_t num_steps;    
            int64_t step;
        
        public:
            RolloutStorage(int64_t num_steps,
                        int64_t num_processes,
                        c10::ArrayRef<int64_t> obs_shape,
                        ActionSpace action_space,
                        int64_t hidden_state_size,
                        torch::Device device);

    };
}


