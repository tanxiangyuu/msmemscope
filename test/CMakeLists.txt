find_package(Python3 REQUIRED COMPONENTS Development)
if (NOT Python3_FOUND)
    message(FATAL_ERROR "Python3 not found.")
else()
    message(STATUS "Found Python3: ${Python3_INCLUDE_DIRS}")
endif()

add_executable(leaks_test)
add_subdirectory(framework)
add_subdirectory(utility)
add_subdirectory(analysis)
add_subdirectory(event_trace)
if (${FUZZ_TESTS})
    add_subdirectory(fuzz_test)
endif()

set(leaks_header ../csrc
                 ../csrc/analysis
                 ../csrc/framework
                 ../csrc/event_trace
                 ../csrc/utility
                 ../platform/securec/include)
set(CMAKE_CXX_FLAGS "-fprofile-arcs -ftest-coverage -lgcov")
set(gtest_header ../opensource/googletest/googletest/include/)

file(GLOB_RECURSE TEST_ORIGINAL_SOURCE ../csrc/framework/*.cpp
                                       ../csrc/analysis/*.cpp
                                       ../csrc/event_trace/*.cpp
                                       ../csrc/utility/*.cpp
                                       ../platform/securec/src/*.c)
list(FILTER TEST_ORIGINAL_SOURCE EXCLUDE REGEX "host_memory_hooks\\.cpp$")
list(FILTER TEST_ORIGINAL_SOURCE EXCLUDE REGEX ".*\/atb_hooks\/.*")

list(SORT TEST_ORIGINAL_SOURCE)
file(GLOB_RECURSE TEST_SOURCE ./test.cpp)
list(SORT TEST_SOURCE)
target_sources(leaks_test PRIVATE ${TEST_SOURCE})
target_sources(leaks_test PRIVATE ${TEST_ORIGINAL_SOURCE})

include_directories(leaks_test ${leaks_header} ${gtest_header} ${Python3_INCLUDE_DIRS})

target_link_libraries(leaks_test gtest_main dl pthread stdc++fs ${Python3_LIBRARIES} rt)
 
add_test(NAME leaks_test COMMAND leaks_test)