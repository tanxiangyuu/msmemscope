set(FUNC_INJECTION_CORE_DIR ${FUNC_INJECTION_DIR}/host_injection/core)

file(GLOB SOURCE "*.cpp, *.h"
${FRAMEWORK_INC_DIR}/client_process.cpp
${FUNC_INJECTION_CORE_DIR}/Communication.cpp
${FUNC_INJECTION_CORE_DIR}/DomainSocket.cpp
${FUNC_HOSTINJECTION_DIR}/utils/FileSystem.cpp
${FUNC_HOSTINJECTION_DIR}/utils/Ustring.cpp
${EVENT_TRACE_INC_DIR}/describe_trace.cpp
)

list(SORT SOURCE)
target_sources(ascend_leaks PRIVATE ${SOURCE})

# libascend_hal.so placeholder for link
add_library(ascend_hal SHARED ${EVENT_TRACE_INC_DIR}/ascend_hal_stub.cpp)
add_subdirectory(hal_hooks)
add_subdirectory(host_memory_hooks)
add_subdirectory(atb_hooks)
add_subdirectory(mstx_hooks)
add_subdirectory(kernel_hooks)
add_subdirectory(memory_pool_trace)
add_subdirectory(op_watch)