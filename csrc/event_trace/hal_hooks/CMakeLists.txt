add_library(leaks_ascend_hal_hook SHARED hal_hooks.cpp)

target_sources(leaks_ascend_hal_hook PRIVATE
  ${EVENT_TRACE_INC_DIR}/event_report.cpp
  ${FRAMEWORK_INC_DIR}/client_process.cpp
  ${FUNC_INJECTION_CORE_DIR}/Communication.cpp
  ${FUNC_INJECTION_CORE_DIR}/DomainSocket.cpp
  ${UTILITY_INC_DIR}/log.cpp
  ${UTILITY_INC_DIR}/call_stack.cpp
  ${UTILITY_INC_DIR}/cpython.cpp
  ${FUNC_HOSTINJECTION_DIR}/utils/FileSystem.cpp
  ${FUNC_HOSTINJECTION_DIR}/utils/Ustring.cpp
  ${EVENT_TRACE_INC_DIR}/trace_manager/event_trace_manager.cpp
)

target_include_directories(leaks_ascend_hal_hook PUBLIC
  ${FRAMEWORK_INC_DIR}
  ${EVENT_TRACE_INC_DIR}
  ${FUNC_INJECTION_DIR}
  ${FUNC_HOSTINJECTION_DIR}
  ${UTILITY_INC_DIR}
  ${SECUREC_INC_DIR}
  ${Python3_INCLUDE_DIRS}
)
target_link_options(leaks_ascend_hal_hook PRIVATE "-Wl,--disable-new-dtags")
target_link_libraries(leaks_ascend_hal_hook ascend_hal stdc++fs c_sec)