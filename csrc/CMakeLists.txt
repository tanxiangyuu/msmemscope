set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/../output)

add_library(ascend_leaks SHARED dummy.cpp)

target_include_directories(
    ascend_leaks PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(framework)
add_subdirectory(event_trace)
add_subdirectory(analysis)

add_executable(leaks ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_link_libraries(leaks ascend_leaks)

set_target_properties(leaks PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_INSTALL_PREFIX}/bin")