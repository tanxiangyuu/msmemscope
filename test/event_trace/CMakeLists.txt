file(GLOB_RECURSE TEST_SOURCE "*.cpp")
list(SORT TEST_SOURCE)
target_sources(memscope_test PRIVATE ${TEST_SOURCE})