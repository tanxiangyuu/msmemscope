add_library(ascend_kernel_hook SHARED runtime_hooks.cpp acl_hooks.cpp)

find_package(Python3 REQUIRED COMPONENTS Development)
if (NOT Python3_FOUND)
    message(FATAL_ERROR "Python3 is no found.")
endif()

target_sources(ascend_kernel_hook PRIVATE
  ${EVENT_TRACE_INC_DIR}/event_report.cpp
  ${EVENT_TRACE_INC_DIR}/kernel_hooks/kernel_event_trace.cpp
  ${EVENT_TRACE_INC_DIR}/kernel_hooks/driver_prof_api.cpp
  ${EVENT_TRACE_INC_DIR}/kernel_hooks/runtime_prof_api.cpp
  ${EVENT_TRACE_INC_DIR}/describe_trace.cpp
  ${EVENT_TRACE_INC_DIR}/vallina_symbol.cpp
  ${FRAMEWORK_INC_DIR}/client_process.cpp
  ${FRAMEWORK_INC_DIR}/client_parser.cpp
  ${FUNC_INJECTION_CORE_DIR}/Communication.cpp
  ${FUNC_INJECTION_CORE_DIR}/DomainSocket.cpp
  ${UTILITY_INC_DIR}/cpython.cpp
  ${UTILITY_INC_DIR}/log.cpp
  ${UTILITY_INC_DIR}/file.cpp
  ${UTILITY_INC_DIR}/sqlite_loader.cpp
  ${UTILITY_INC_DIR}/ustring.cpp
  ${UTILITY_INC_DIR}/call_stack.cpp
  ${UTILITY_INC_DIR}/cpython.cpp
  ${UTILITY_INC_DIR}/path.cpp
  ${UTILITY_INC_DIR}/ustring.cpp
  ${FUNC_HOSTINJECTION_DIR}/utils/FileSystem.cpp
  ${FUNC_HOSTINJECTION_DIR}/utils/Ustring.cpp
  ${EVENT_TRACE_INC_DIR}/trace_manager/event_trace_manager.cpp
)

target_include_directories(ascend_kernel_hook PUBLIC
  ${Python3_INCLUDE_DIRS}
  ${FRAMEWORK_INC_DIR}
  ${EVENT_TRACE_INC_DIR}
  ${FUNC_INJECTION_DIR}
  ${FUNC_HOSTINJECTION_DIR}
  ${UTILITY_INC_DIR}
  ${SECUREC_INC_DIR}
)
target_link_options(ascend_kernel_hook PRIVATE "-Wl,--disable-new-dtags")
target_link_libraries(ascend_kernel_hook ascend_hal dl pthread stdc++fs c_sec)

target_sources(ascend_leaks PRIVATE
  driver_prof_api.cpp
  kernel_event_trace.cpp
  runtime_prof_api.cpp
)
