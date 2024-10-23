set(FUNC_INJECTION_CORE_DIR ${FUNC_INJECTION_DIR}/host_injection/core)

file(GLOB SOURCE "*.cpp"
${FUNC_INJECTION_CORE_DIR}/RemoteProcess.cpp
${FUNC_INJECTION_CORE_DIR}/Communication.cpp
${FUNC_INJECTION_CORE_DIR}/DomainSocket.cpp)
list(SORT SOURCE)

target_sources(ascend_leaks PRIVATE ${SOURCE})