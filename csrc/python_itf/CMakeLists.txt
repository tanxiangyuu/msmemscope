add_library(_msleaks SHARED)
set_target_properties(_msleaks PROPERTIES PREFIX "")
find_package(Python3 REQUIRED COMPONENTS Development)
if (NOT Python3_FOUND)
    message(FATAL_ERROR "Python3 is no found.")
endif()
target_include_directories(_msleaks PUBLIC
  ${Python3_INCLUDE_DIRS}
)

file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
target_sources(_msleaks PRIVATE ${SOURCES})
target_link_libraries(_msleaks PUBLIC ascend_leaks)
