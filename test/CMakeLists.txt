add_executable(leaks_test)

set(leaks_header ../csrc/analysis
                 ../csrc/framework
                 ../csrc/event_trace)
set(CMAKE_CXX_FLAGS "-fprofile-arcs -ftest-coverage -lgcov")
set(gtest_header ../opensource/googletest/googletest/include/)

file(GLOB_RECURSE TEST_ORIGINAL_SOURCE ../csrc/framework/*.cpp
                                       ../csrc/analysis/*.cpp
                                       ../csrc/event_trace/*.cpp)

list(SORT TEST_ORIGINAL_SOURCE)
file(GLOB_RECURSE TEST_SOURCE ./test.cpp)
list(SORT TEST_SOURCE)
target_sources(leaks_test PRIVATE ${TEST_SOURCE})
target_sources(leaks_test PRIVATE ${TEST_ORIGINAL_SOURCE})

include_directories(leaks_test ${leaks_header} ${gtest_header})
target_link_libraries(leaks_test ascend_leaks gtest_main)
 
add_test(NAME leaks_test COMMAND leaks_test)