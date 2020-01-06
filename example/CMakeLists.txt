add_executable(environment environment.cpp communicator.cpp)

target_link_libraries(environment PRIVATE drift)
