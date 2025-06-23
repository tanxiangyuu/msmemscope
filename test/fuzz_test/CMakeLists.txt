project(leaks_fuzz)

add_executable(leaks_fuzz)
file(GLOB_RECURSE SOURCE "*.cpp")
list(SORT SOURCE)
target_sources(leaks_fuzz PRIVATE ${SOURCE})

get_filename_component(SRC_DIR ${ROOT_DIR}/csrc ABSOLUTE)

set(leaks_header ${SRC_DIR}
                 ${SRC_DIR}/analysis
                 ${SRC_DIR}/framework
                 ${SRC_DIR}/event_trace
                 ${SRC_DIR}/utility
                 ${ROOT_DIR}/platform/func_injection
                 ${ROOT_DIR}/platform/func_injection/host_injection
                 ${ROOT_DIR}/platform/securec/include)

set(gtest_header ../opensource/googletest/googletest/include/)

file(GLOB_RECURSE SRC_SOURCE ${SRC_DIR}/framework/*.cpp
                                       ${SRC_DIR}/analysis/*.cpp
                                       ${SRC_DIR}/event_trace/*.cpp
                                       ${SRC_DIR}/utility/*.cpp
                                       ${ROOT_DIR}/platform/func_injection/host_injection/core/Communication.cpp
                                       ${ROOT_DIR}/platform/func_injection/host_injection/core/DomainSocket.cpp
                                       ${ROOT_DIR}/platform/func_injection/host_injection/core/LocalProcess.cpp
                                       ${ROOT_DIR}/platform/func_injection/host_injection/utils/FileSystem.cpp
                                       ${ROOT_DIR}/platform/func_injection/host_injection/utils/Ustring.cpp
                                       ${ROOT_DIR}/platform/func_injection/host_injection/utils/FuncSelector.cpp
                                       ${ROOT_DIR}/platform/securec/src/*.c)
list(FILTER SRC_SOURCE EXCLUDE REGEX "host_memory_hooks\\.cpp$")
list(FILTER SRC_SOURCE EXCLUDE REGEX ".*\/atb_hooks\/.*")
list(SORT SRC_SOURCE)
target_sources(leaks_fuzz PRIVATE ${SRC_SOURCE})

add_compile_options(
    -g
    -fPIC
    -Wall
    -Winvalid-pch
    -fms-extensions
    -std=c++11
    -O0
    -fno-omit-frame-pointer
    -ftest-coverage
    -fprofile-arcs
    -fdump-rtl-expand
    -Wno-pmf-conversions
    -D_GLIBCXX_USE_CXX11_ABI=0
    # for asan
    -fsanitize=address
    -fsanitize=undefined
    -fsanitize-coverage=trace-cmp
    -fsanitize-coverage=trace-pc
)

add_link_options(
    -fsanitize=address
    -fsanitize=undefined
    -fsanitize-coverage=trace-pc,trace-cmp
)

target_link_directories(leaks_fuzz PUBLIC
    ${ROOT_DIR}/opensource/Secodefuzz/examples/out-bin-x64/)


target_link_libraries(leaks_fuzz gtest gtest_main pthread c_sec Secodefuzz stdc++fs ${Python3_LIBRARIES} ${SQLite3_LIBRARIES})
target_include_directories(leaks_fuzz PUBLIC
    ${ROOT_DIR}/opensource/Secodefuzz/Secodefuzz
    ${gtest_header}
    ${leaks_header}
    ${Python3_INCLUDE_DIRS}
    ${SQLite3_INCLUDE_DIRS})