set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/../output)
find_package(Python3 REQUIRED COMPONENTS Development)

if (NOT Python3_FOUND)
    message(FATAL_ERROR "Python3 is not found.")
else()
    message(STATUS "csrc Found Python3: ${Python3_INCLUDE_DIRS}")
endif()

get_filename_component(SECUREC_INC_DIR ../platform/securec/include ABSOLUTE)
get_filename_component(FRAMEWORK_INC_DIR ./framework ABSOLUTE)
get_filename_component(EVENT_TRACE_INC_DIR ./event_trace ABSOLUTE)
get_filename_component(UTILITY_INC_DIR ./utility ABSOLUTE)

add_library(ascend_leaks SHARED dummy.cpp)

target_include_directories(
    ascend_leaks PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                        ${FRAMEWORK_INC_DIR}
                        ${SECUREC_INC_DIR}
                        ${Python3_INCLUDE_DIRS}
                        ${EVENT_TRACE_INC_DIR}
)

#避免依赖python版本
target_link_options(ascend_leaks PRIVATE "-Wl,--copy-dt-needed-entries")
target_link_libraries(ascend_leaks pthread stdc++fs dl c_sec rt)

add_subdirectory(framework)
add_subdirectory(utility)
add_subdirectory(event_trace)
add_subdirectory(analysis)
add_subdirectory(python_itf)

add_executable(msleaks ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_link_libraries(msleaks ascend_leaks)

set_target_properties(msleaks PROPERTIES
    OUTPUT_NAME "msleaks.bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_INSTALL_PREFIX}/bin"
)
set_target_properties(leaks_ascend_hal_hook host_memory_hook ascend_mstx_hook ascend_kernel_hook atb_abi_0_hook atb_abi_1_hook ascend_leaks _msleaks PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_INSTALL_PREFIX}/lib64"
)