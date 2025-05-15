add_library(ascend_mstx_hook SHARED mstx_manager.cpp mstx_inject.cpp aten_manager.cpp)

target_sources(ascend_mstx_hook PRIVATE
  ${EVENT_TRACE_INC_DIR}/event_report.cpp
  ${EVENT_TRACE_INC_DIR}/memory_pool_trace/memory_pool_trace_manager.cpp
  ${EVENT_TRACE_INC_DIR}/memory_pool_trace/atb_memory_pool_trace.cpp
  ${FRAMEWORK_INC_DIR}/client_process.cpp
  ${UTILITY_INC_DIR}/log.cpp
  ${UTILITY_INC_DIR}/call_stack.cpp
  ${UTILITY_INC_DIR}/cpython.cpp
)

target_include_directories(ascend_mstx_hook PUBLIC
  ${FRAMEWORK_INC_DIR}
  ${EVENT_TRACE_INC_DIR}
  ${FUNC_INJECTION_DIR}
  ${FUNC_HOSTINJECTION_DIR}
  ${UTILITY_INC_DIR}
  ${SECUREC_INC_DIR}
  ${Python3_INCLUDE_DIRS}
)

target_link_options(ascend_mstx_hook PRIVATE "-Wl,--disable-new-dtags")
target_link_libraries(ascend_mstx_hook c_sec)