cmake_minimum_required(VERSION 3.16)
 
project(msleaks VERSION 0.1.0)
 
set(CMAKE_VERBOSE_MAKEFILE ON)
 
# 安全规范编译命令
add_compile_options("-fPIE")
add_compile_options("-fPIC")
add_compile_options("-fstack-protector-all")
add_compile_options("-Wall")
add_compile_options("-D_FORTIFY_SOURCE=2")
add_compile_options("-fno-strict-aliasing")
if (${BUILD_TESTS})
  add_compile_options("-O0") #内联扩展导致lcov覆盖率扫描不到，因此test工程不进行编译优化
else()
  add_compile_options("-O2")
endif()
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
add_link_options("-Wl,-z,now")
add_link_options("-Wl,-z,relro")
add_link_options("-Wl,-z,noexecstack")
add_link_options("-pie")
add_link_options("-s") #strip
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

option(BUILD_TESTS "build unit tests" OFF)

# 设置git commit ID
find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
  OUTPUT_VARIABLE MSLEAKS_COMMIT_ID OUTPUT_STRIP_TRAILING_WHITESPACE)
add_definitions(-D__MSLEAKS_COMMIT_ID__="${MSLEAKS_COMMIT_ID}")

add_subdirectory(platform)
add_subdirectory(csrc)

if (${BUILD_TESTS})
  message(STATUS "enable unit tests build")
  enable_testing()
  add_subdirectory(opensource)
  add_subdirectory(test)
endif()