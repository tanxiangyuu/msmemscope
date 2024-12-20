set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/../output)
set(FUNC_INJECTION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../platform/func_injection/)
set(FUNC_HOSTINJECTION_DIR ${FUNC_INJECTION_DIR}/host_injection/)
get_filename_component(SECUREC_INC_DIR ../platform/securec/include ABSOLUTE)
get_filename_component(FRAMEWORK_INC_DIR ./framework ABSOLUTE)
get_filename_component(EVENT_TRACE_INC_DIR ./event_trace ABSOLUTE)
get_filename_component(UTILITY_INC_DIR ./utility ABSOLUTE)

add_library(ascend_leaks SHARED dummy.cpp)

target_include_directories(
    ascend_leaks PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                        ${FUNC_INJECTION_DIR}
                        ${FRAMEWORK_INC_DIR}
                        ${FUNC_HOSTINJECTION_DIR}
                        ${SECUREC_INC_DIR}
)

target_link_libraries(ascend_leaks pthread stdc++fs dl c_sec)

add_subdirectory(framework)
add_subdirectory(utility)
add_subdirectory(event_trace)
add_subdirectory(analysis)

add_executable(leaks ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_link_libraries(leaks ascend_leaks)

set_target_properties(leaks PROPERTIES
    OUTPUT_NAME "leaks.bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_INSTALL_PREFIX}/bin"
)
set_target_properties(ascend_hal_hook ascend_mstx_hook ascend_kernel_hook leaks_torch_npu_trace PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_INSTALL_PREFIX}/lib64"
)