file(GLOB SOURCE "*.cpp")
list(SORT SOURCE)
list(REMOVE_ITEM SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/call_stack.cpp")
target_sources(ascend_leaks PRIVATE ${SOURCE})