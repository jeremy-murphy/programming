find_package(GTest)

if (GTEST_FOUND)
    enable_testing()

    include_directories(${GTEST_INCLUDE_DIRS})

    add_executable(test_functional test_functional.cpp)
    target_link_libraries(test_functional ${GTEST_BOTH_LIBRARIES})
    add_test(functional test_functional)
    
    add_executable(test_find test_find.cpp)
    target_link_libraries(test_find ${GTEST_BOTH_LIBRARIES})
    add_test(find test_find)
    
endif()
