if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sqlite3.c")
    message(FATAL_ERROR "sqlite3.c not found in ${CMAKE_CURRENT_SOURCE_DIR}. Please run download script.")
endif()

add_library(sqlite3 STATIC
    sqlite3.c
)

target_include_directories(sqlite3 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

# 可选：优化配置（根据你的需求调整）
target_compile_definitions(sqlite3 PRIVATE
    SQLITE_THREADSAFE=0
    SQLITE_DEFAULT_MEMSTATUS=0
    SQLITE_OMIT_LOAD_EXTENSION
)

# ========== 新增：调试打印 ==========
message(STATUS "sqlite3 库已创建，头文件路径：${CMAKE_CURRENT_SOURCE_DIR}")