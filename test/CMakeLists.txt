add_executable(leaks_test)
add_subdirectory(framework)
add_subdirectory(utility)
add_subdirectory(analysis)
add_subdirectory(event_trace)

set(leaks_header ../csrc/analysis
                 ../csrc/framework
                 ../csrc/event_trace
                 ../platform/securec/include)
set(CMAKE_CXX_FLAGS "-fprofile-arcs -ftest-coverage -lgcov")
set(gtest_header ../opensource/googletest/googletest/include/)

file(GLOB_RECURSE TEST_ORIGINAL_SOURCE ../csrc/framework/*.cpp
                                       ../csrc/analysis/*.cpp
                                       ../csrc/event_trace/*.cpp)
list(FILTER TEST_ORIGINAL_SOURCE EXCLUDE REGEX "host_memory_hooks\\.cpp$")

list(SORT TEST_ORIGINAL_SOURCE)
file(GLOB_RECURSE TEST_SOURCE ./test.cpp)
list(SORT TEST_SOURCE)
target_sources(leaks_test PRIVATE ${TEST_SOURCE})
target_sources(leaks_test PRIVATE ${TEST_ORIGINAL_SOURCE})

include_directories(leaks_test ${leaks_header} ${gtest_header})
target_link_libraries(leaks_test ascend_leaks c_sec gtest_main dl)
 
add_test(NAME leaks_test COMMAND leaks_test)