#pragma once

#include <vector>
#include <torch/torch.h>

namespace drift {
    
    /*
     * Running mean provides a baseline for model performance
     */ 
    class RunningMeanStdImpl : public torch::nn::Module {
        private:
            torch::Tensor count, mean, variance;

            void update_from_moments(torch::Tensor batch_mean,
                                     torch::Tensor batch_variance,
                                     torch::Tensor batch_count);
            
        public:
            // std over size
            explicit RunningMeanStdImpl(int size);
            RunningMeanStdImpl(std::vector<float> mean, std::vector<float> variance);
            
            // We must update the running attributes given a new observation
            void update(torch::Tensor observation);

            inline int get_count() const { return static_cast<int>(count.item().toFloat()); }
            inline torch::Tensor get_mean() const { return mean.clone(); }
            inline torch::Tensor get_variance() const { return variance.clone(); }
            inline void set_cout(int count) { this->count[0] = count + 1e-8; }
    };
    
    // macro to define a torch module over the running mean and std
    TORCH_MODULE(RunningMeanStd);

}
