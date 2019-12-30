target_sources(drift PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/a2c.cpp}
    )

if (DRIFT_BUILD_TESTS)
    target_sources(drift_tests PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/a2c.cpp
    )
endif(DRIFT_BUILD_TESTS)


