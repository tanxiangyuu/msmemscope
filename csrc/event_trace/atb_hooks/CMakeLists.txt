add_library(atb_abi_0_hook SHARED atb_abi_0_hooks.cpp)

target_sources(atb_abi_0_hook PRIVATE
  atb_hooks.cpp
  enum2string.cpp
  ${EVENT_TRACE_INC_DIR}/op_watch/op_excute_watch.cpp
  ${EVENT_TRACE_INC_DIR}/op_watch/tensor_dumper.cpp
  ${EVENT_TRACE_INC_DIR}/op_watch/tensor_monitor.cpp
  ${UTILITY_INC_DIR}/log.cpp
  ${UTILITY_INC_DIR}/file.cpp
  ${UTILITY_INC_DIR}/calculate_data_check_sum.cpp
  ${UTILITY_INC_DIR}/path.cpp
  ${FRAMEWORK_INC_DIR}/client_process.cpp
)

target_compile_definitions(atb_abi_0_hook PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)

target_include_directories(atb_abi_0_hook PUBLIC
  ${FRAMEWORK_INC_DIR}
  ${EVENT_TRACE_INC_DIR}
  ${FUNC_INJECTION_DIR}
  ${FUNC_HOSTINJECTION_DIR}
  ${UTILITY_INC_DIR}
  ${SECUREC_INC_DIR}
)

target_link_libraries(atb_abi_0_hook dl)


add_library(atb_abi_1_hook SHARED atb_abi_1_hooks.cpp)

target_compile_options(atb_abi_1_hook PRIVATE
  -U_GLIBCXX_USE_CXX11_ABI
  -D_GLIBCXX_USE_CXX11_ABI=1
)

target_sources(atb_abi_1_hook PRIVATE
  atb_hooks.cpp
  enum2string.cpp
  ${EVENT_TRACE_INC_DIR}/op_watch/op_excute_watch.cpp
  ${EVENT_TRACE_INC_DIR}/op_watch/tensor_dumper.cpp
  ${EVENT_TRACE_INC_DIR}/op_watch/tensor_monitor.cpp
  ${UTILITY_INC_DIR}/log.cpp
  ${UTILITY_INC_DIR}/file.cpp
  ${UTILITY_INC_DIR}/calculate_data_check_sum.cpp
  ${UTILITY_INC_DIR}/path.cpp
  ${FRAMEWORK_INC_DIR}/client_process.cpp
)

target_include_directories(atb_abi_1_hook PUBLIC
  ${FRAMEWORK_INC_DIR}
  ${EVENT_TRACE_INC_DIR}
  ${FUNC_INJECTION_DIR}
  ${FUNC_HOSTINJECTION_DIR}
  ${UTILITY_INC_DIR}
  ${SECUREC_INC_DIR}
)

target_link_libraries(atb_abi_1_hook dl)