// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef LEAKS_UTILITY_UTILS_H
#define LEAKS_UTILITY_UTILS_H

#include <unistd.h>
#include <syscall.h>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <limits>
#include <typeinfo>
#include <iostream>
#include <unistd.h>

namespace Utility {
    inline uint64_t GetTid()
    {
        static thread_local uint64_t tid = static_cast<uint64_t>(syscall(SYS_gettid));
        return tid;
    }
    inline uint64_t GetPid()
    {
        static thread_local uint64_t pid = static_cast<uint64_t>(getpid());
        return pid;
    }

    inline uint64_t GetTimeMicroseconds()
    {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
        return static_cast<uint64_t>(duration.count());
    }

    inline uint64_t GetTimeNanoseconds()
    {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());
        return static_cast<uint64_t>(duration.count());
    }
    
    // 多线程情况下调用，需加锁保护
    inline std::string GetDateStr()
    {
        auto now = std::chrono::system_clock::now();
        std::time_t time = std::chrono::system_clock::to_time_t(now);
        std::tm localTime = *std::localtime(&time);
        std::ostringstream oss;
        oss << std::put_time(&localTime, "%Y%m%d%H%M%S");
        return oss.str();
    }

    // a + b
    template <typename T>
    T inline GetAddResult(T &a, T &b)
    {
        if ((b > 0 && a > std::numeric_limits<T>::max() - b) ||
            (b < 0 && a < std::numeric_limits<T>::min() - b)) {
            std::cout << "Add overflow:" << typeid(T).name() << a << " " << b << std::endl;
            return a;
        }
        return a + b;
    }

    // a - b
    template <typename T>
    T inline GetSubResult(T &a, T &b)
    {
        if ((b > 0 && a < std::numeric_limits<T>::min() + b) ||
            (b < 0 && a > std::numeric_limits<T>::max() + b)) {
            std::cout << "Sub overflow:" << typeid(T).name() << a << " " << b << std::endl;
            return a;
        }
        return a - b;
    }

    inline bool StrToInt64(int64_t &dest, const std::string &str)
    {
        if (str.empty()) {
            return false;
        }
        size_t pos = 0;
        try {
            dest = std::stoll(str, &pos);
        } catch (...) {
            return false;
        }
        if (pos != str.size()) {
            return false;
        }
        return true;
    }

    inline bool StrToUint64(uint64_t &dest, const std::string &str)
    {
        if (str.empty()) {
            return false;
        }
        size_t pos = 0;
        try {
            dest = std::stoull(str, &pos);
        } catch (...) {
            return false;
        }
        if (pos != str.size()) {
            return false;
        }
        return true;
    }

    inline bool StrToUint32(uint32_t &dest, const std::string &str)
    {
        if (str.empty()) {
            return false;
        }
        size_t pos = 0;
        try {
            unsigned long value = std::stoul(str, &pos);
            if (pos == str.size() && value <= std::numeric_limits<uint32_t>::max()) {
                dest = static_cast<uint32_t>(value);
                return true;
            }
        } catch (...) {
            return false;
        }
        return false;
    }
}

#endif