// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef LEAKS_UTILITY_UTILS_H
#define LEAKS_UTILITY_UTILS_H

#include <syscall.h>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace Utility {
    static uint64_t GetTid()
    {
        static thread_local uint64_t tid = static_cast<uint64_t>(syscall(SYS_gettid));
        return tid;
    }
    static uint64_t GetPid()
    {
        static thread_local uint64_t pid = static_cast<uint64_t>(getpid());
        return pid;
    }
    static uint64_t GetTimeMicroseconds()
    {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
        return static_cast<uint64_t>(duration.count());
    }
    static std::string GetDateStr()
    {
        auto now = std::chrono::system_clock::now();
        std::time_t time = std::chrono::system_clock::to_time_t(now);
        std::tm localTime = *std::localtime(&time);
        std::ostringstream oss;
        oss << std::put_time(&localTime, "%Y%m%d%H%M%S");
        return oss.str();
    }
}

#endif