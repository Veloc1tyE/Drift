#pragma once

#include <string>
#include <torch/torch.h>

namespace cpprl  {

    /**
     * Define an action space which contains a type of action
     * the agent can perform, as well as a vector containing integers
     * which encodes information for the action
     */ 
    
    struct ActionSpace  {
        std::string type;
        std::vector<int64_t> shape;
    };
}
