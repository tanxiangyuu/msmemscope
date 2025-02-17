add_library(host_memory_hook SHARED host_memory_hooks.cpp)
 
target_sources(host_memory_hook PRIVATE
  ${EVENT_TRACE_INC_DIR}/event_report.cpp
  ${FUNC_INJECTION_CORE_DIR}/LocalProcess.cpp
  ${FUNC_INJECTION_CORE_DIR}/Communication.cpp
  ${FUNC_INJECTION_CORE_DIR}/DomainSocket.cpp
  ${UTILITY_INC_DIR}/log.cpp
  ${FUNC_HOSTINJECTION_DIR}/utils/FileSystem.cpp
  ${FUNC_HOSTINJECTION_DIR}/utils/Ustring.cpp
)
 
target_include_directories(host_memory_hook PUBLIC
  ${FRAMEWORK_INC_DIR}
  ${EVENT_TRACE_INC_DIR}
  ${FUNC_INJECTION_DIR}
  ${FUNC_HOSTINJECTION_DIR}
  ${UTILITY_INC_DIR}
  ${SECUREC_INC_DIR}
)
 
target_link_libraries(host_memory_hook dl)