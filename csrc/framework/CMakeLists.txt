set(FUNC_INJECTION_CORE_DIR ${FUNC_INJECTION_DIR}/host_injection/core)

file(GLOB_RECURSE SOURCE "*.cpp"
${FUNC_INJECTION_CORE_DIR}/RemoteProcess.cpp
${FUNC_INJECTION_CORE_DIR}/Communication.cpp
${FUNC_INJECTION_CORE_DIR}/DomainSocket.cpp)
list(SORT SOURCE)

target_sources(ascend_leaks PRIVATE ${SOURCE})
target_include_directories(ascend_leaks PUBLIC
  ${UTILITY_INC_DIR}
)