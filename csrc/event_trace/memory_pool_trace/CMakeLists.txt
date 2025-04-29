file(GLOB SOURCE "*.cpp")
list(SORT SOURCE)
target_sources(ascend_leaks PRIVATE ${SOURCE})
target_sources(ascend_leaks PRIVATE
  ${UTILITY_INC_DIR}/call_stack.cpp
  ${UTILITY_INC_DIR}/cpython.cpp
)