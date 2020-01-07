#pragma once

#include <string>
#include <vector>

#include "cpprl/storage.h"

namespace cpprl {
    
    /*
     * Abstractions for use in learning algorithms
     */ 

    struct UpdateDatum {
        std::string name;
        float value;
    };
    
    /*
     * A2C inherits from Algorithm and requires the constructs to update based on rollouts
     */

    class Algorithm {
        public:
            virtual ~Algorithm() = 0;
            virtual std::vector<UpdateDatum> update(RolloutStorage &rollouts, float decay_level = 1) = 0;
    };

    inline Algorithm::~Algorithm() {}
}


