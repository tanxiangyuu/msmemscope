add_library(leaks_torch_npu_trace SHARED torch_npu_trace.cpp)
set(TORCH_NPU_TRACE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FUNC_INJECTION_CORE_DIR ${FUNC_INJECTION_DIR}/host_injection/core)

target_sources(leaks_torch_npu_trace PRIVATE
  ${EVENT_TRACE_INC_DIR}/event_report.cpp
  ${FUNC_INJECTION_CORE_DIR}/LocalProcess.cpp
  ${FUNC_INJECTION_CORE_DIR}/Communication.cpp
  ${FUNC_INJECTION_CORE_DIR}/DomainSocket.cpp
  ${UTILITY_INC_DIR}/log.cpp
  ${FUNC_HOSTINJECTION_DIR}/utils/FileSystem.cpp
  ${FUNC_HOSTINJECTION_DIR}/utils/Ustring.cpp
)
target_include_directories(leaks_torch_npu_trace PUBLIC
  ${FRAMEWORK_INC_DIR}
  ${EVENT_TRACE_INC_DIR}
  ${FUNC_INJECTION_DIR}
  ${FUNC_HOSTINJECTION_DIR}
  ${UTILITY_INC_DIR}
  ${TORCH_NPU_TRACE_DIR}
)