cmake_minimum_required(VERSION 3.0)
 
project(leaks VERSION 0.1.0)
 
set(CMAKE_VERBOSE_MAKEFILE ON)
 
# 安全规范编译命令
add_compile_options("-fPIE")
add_compile_options("-std=c++11")
add_compile_options("-fPIC")
add_compile_options("-fstack-protector-all")
add_compile_options("-Wall")
add_compile_options("-D_FORTIFY_SOURCE=2")
add_compile_options("-fno-strict-aliasing")
add_compile_options("-O2")

add_link_options("-Wl,-z,now")
add_link_options("-Wl,-z,relro")
add_link_options("-Wl,-z,noexecstack")
add_link_options("-pie")
add_link_options("-s") #strip

option(BUILD_TESTS "build unit tests" OFF)

add_subdirectory(platform)
add_subdirectory(csrc)

if (${BUILD_TESTS})
  message(STATUS "enable unit tests build")
  enable_testing()
  add_subdirectory(opensource)
  add_subdirectory(test)
endif()
