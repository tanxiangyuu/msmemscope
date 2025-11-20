file(GLOB TEST_SOURCE "*.cpp")
list(SORT TEST_SOURCE)

target_sources(memscope_test PRIVATE ${TEST_SOURCE})