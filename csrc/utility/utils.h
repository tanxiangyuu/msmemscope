// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef LEAKS_UTILITY_UTILS_H
#define LEAKS_UTILITY_UTILS_H

#include <syscall.h>

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
}

#endif