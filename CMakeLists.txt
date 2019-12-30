cmake_minimum_required(VERSION 3.10.0)

project(Drift
        Languages CXX
        VERSION 1.2.0
        Description "Reinforcement Learning environment for fusion simulation"
    )

# Project-wide properties
set(CMAKE_CXX_STANDARD 17)

# Cppcheck
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    list(APPEND CPPCHECK_ARGS
        --enable=warning
        --std=c++14
        --force
        --verbose
        --quiet
        --inline-suppr
        --error-exitcode=1
        --language=c++
        -i${CMAKE_CURRENT_LIST_DIR}/example/lib
        -I ${CMAKE_CURRENT_LIST_DIR}/src 
        -I ${CMAKE_CURRENT_LIST_DIR}/include 
        -I ${CMAKE_CURRENT_LIST_DIR}/example 
        ${CMAKE_CURRENT_LIST_DIR}/src
        ${CMAKE_CURRENT_LIST_DIR}/example
    )

    add_custom_target(
        check
        COMMAND cppcheck ${CPPCHECK_ARGS}
        COMMENT "Running Cppcheck"
    )
endif(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)

# Dependencies
# PyTorch
if (NOT TORCH_FOUND)
    find_package(Torch REQUIRED)
    if (TORCH_CXX_FLAGS)
        set(CMAKE_CXX_FLAGS ${TORCH_CXX_FLAGS})
    endif()
endif (NOT TORCH_FOUND)

# Define targets
add_library(drift STATIC "")
target_compile_definitions(drift PRIVATE DOCTEST_CONFIG_DISABLE)
option(DRIFT_BUILD_TESTS "Whether o not to build Drift tests" ON)
if (DRIFT_BUILD_TESTS)
    add_executable(drift_tests)
endif(DRIFT_BUILD_TESTS)

# Enable all warnings
if(MSVC)
    target_compile_options(drift PRIVATE /W0)
else(MSVC)
    target_compile_options(drift PRIVATE -Wall -Wextra -pedantic)
endif(MSVC)

# Includes
set(DRIFT_INCLUDE_DIRS
    include
    src
    $(TORCH_INCLUDE_DIRS)
)

target_include_directories(drift PRIVATE ${DRIFT_INCLUDE_DIRS})
if (DRIFT_BUILD_TESTS)
    target_include_directories(drift_tests PRIVATE ${DRIFT_INCLUDE_DIRS})
endif(DRIFT_BUILD_TESTS)

# Linking
target_link_libraries(drift torch ${TORCH_LIBRARIES})
target_link_libraries(cpprl torch ${TORCH_LIBRARIES})
if (DRIFT_BUILD_TESTS)
    target_link_libraries(drift_tests PRIVATE ${TORCH_LIBRARIES})
endif(DRIFT_BUILD_TESTS)

# Example
option(DRIFT_BUILD_EXAMPLE "Whether or not to build the example" ON)
if (DRIFT_BUILD_EXAMPLE)
    add_subdirectory(example)
endif(DRIFT_BUILD_EXAMPLE)

# Recurse into source tree
add_subdirectory(src)









