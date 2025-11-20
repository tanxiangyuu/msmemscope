add_library(_msmemscope SHARED)
set_target_properties(_msmemscope PROPERTIES PREFIX "")
find_package(Python3 REQUIRED COMPONENTS Development)
if (NOT Python3_FOUND)
    message(FATAL_ERROR "Python3 is no found.")
endif()
target_include_directories(_msmemscope PUBLIC
  ${Python3_INCLUDE_DIRS}
)

file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
target_sources(_msmemscope PRIVATE ${SOURCES})
target_sources(_msmemscope PRIVATE
  ${EVENT_TRACE_INC_DIR}/python_trace.cpp
  ${EVENT_TRACE_INC_DIR}/describe_trace.cpp
  ${UTILITY_INC_DIR}/ustring.cpp
)
target_link_libraries(_msmemscope PUBLIC ascend_leaks)
