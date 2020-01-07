#pragma once

#include <vector>
#include <torch/torch.h>
#include "cpprl/running_mean_std.h"

namespace cpprl {
    
    /*
     * We must normalise observations to speed up learning
     */ 

    class ObservationNormalizer;

    class ObservationNormalizerImpl : public torch::nn::Module {
        private:
            // attributes of normalising transformation
            torch::Tensor clip;
            RunningMeanStd rms;

        public:
            // explicit implementation for normalising an observation
            explicit ObservationNormalizerImpl(int size, float clip = 10);
            
            /* Implementation for normalising observations
             * given means and variances of outputs in output vector 
             */
            ObservationNormalizerImpl(const std::vector<float> &means,
                                      const std::vector<float> &variances,
                                      float clip = 10.);
            
            // explicit implementation of the normalising process given a vector of normalisers
            // can be useful when we need to normalise different types of observations simultaneously
            explicit ObservationNormalizerImpl(const std::vector<ObservationNormalizer> &others);
                
            
            // Methods for processing and applying the normalisation
            torch::Tensor process_observation(torch::Tensor observation) const;
            std::vector<float> get_mean() const;
            std::vector<float> get_variance() const;

            // normalisation transformation will update after each new observation
            void update(torch::Tensor observations);
            
            // clip observation if too extreme, to prevent outliers
            inline float get_clip_value() const { return clip.item().toFloat(); }

            // retrieve the count for the root-mean-square of observations
            inline float get_step_count() const { return rms->get_count(); } // rms is useful for normalisation
    };
    
    // store the observation normaliser as a torch module, we're keeping track of components
    TORCH_MODULE(ObservationNormalizer);
}
