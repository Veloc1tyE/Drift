set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CPPZMQ_DIR ${LIB_DIR}/msgpack-c)
set(SPDLOG_DIR ${LIB_DIR}/spdlog)
set(ZMQ_DIR ${LIB_DIR}/libzmq)

# Spdlog
add_subdirectory(${SPDLOG_DIR})

target_include_directories(
    PRIVATE
    .
    ../include
    ../lib/spdlog/include
    ${CPPZMQ_DIR}
    ${MSGPACK_DIR}/include
    ${SPDLOG_DIR}/include
)
