add_library(leaks_ascend_hal_hook SHARED hal_hooks.cpp)

target_sources(leaks_ascend_hal_hook PRIVATE
  ${EVENT_TRACE_INC_DIR}/event_report.cpp
  ${UTILITY_INC_DIR}/log.cpp
  ${UTILITY_INC_DIR}/call_stack.cpp
  ${UTILITY_INC_DIR}/cpython.cpp
  ${EVENT_TRACE_INC_DIR}/trace_manager/event_trace_manager.cpp
)

target_include_directories(leaks_ascend_hal_hook PUBLIC
  ${FRAMEWORK_INC_DIR}
  ${EVENT_TRACE_INC_DIR}
  ${UTILITY_INC_DIR}
  ${SECUREC_INC_DIR}
  ${Python3_INCLUDE_DIRS}
  ${ANALYSIS_DIR}
  ${JSON_DIR}
  ${SQLITE3_DIR}
)
target_link_options(leaks_ascend_hal_hook PRIVATE "-Wl,--disable-new-dtags")
target_link_libraries(leaks_ascend_hal_hook ascend_hal stdc++fs c_sec)