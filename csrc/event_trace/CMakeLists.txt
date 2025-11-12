file(GLOB SOURCE "*.cpp, *.h"
${EVENT_TRACE_INC_DIR}/describe_trace.cpp
${EVENT_TRACE_INC_DIR}/vallina_symbol.cpp
)

list(SORT SOURCE)
target_sources(ascend_leaks PRIVATE ${SOURCE})

# libascend_hal.so placeholder for link
add_library(ascend_hal SHARED ${EVENT_TRACE_INC_DIR}/ascend_hal_stub.cpp)
add_subdirectory(hal_hooks)
add_subdirectory(atb_hooks)
add_subdirectory(mstx_hooks)
add_subdirectory(kernel_hooks)
add_subdirectory(memory_pool_trace)
add_subdirectory(memory_watch)
add_subdirectory(trace_manager)