file(GLOB_RECURSE TEST_SOURCE "*.cpp")
list(SORT TEST_SOURCE)
target_sources(leaks_test PRIVATE ${TEST_SOURCE})