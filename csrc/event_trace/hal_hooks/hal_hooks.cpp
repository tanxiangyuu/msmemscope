// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "hal_hooks.h"
#include <string>
#include <dlfcn.h>

/*
#include <cstdlib> // 用于getpid()
#include <thread> // 用于std::this_thread::get_id()
#include <iostream>
#include <sys/types.h>
#include <unistd.h>
*/

#include "event_report.h"
#include "log.h"

using namespace Leaks;

constexpr uint64_t MEM_VIRT_BIT = 10;
constexpr uint64_t MEM_VIRT_WIDTH = 4;
constexpr uint64_t MEM_SVM_VAL = 0x0;
constexpr uint64_t MEM_DEV_VAL = 0x1;
constexpr uint64_t MEM_HOST_VAL = 0x2;
constexpr uint64_t MEM_DVPP_VAL = 0x3;
constexpr uint64_t MEM_HOST = MEM_HOST_VAL << MEM_VIRT_BIT;
constexpr uint64_t MEM_DEV = MEM_DEV_VAL << MEM_VIRT_BIT;
constexpr uint64_t MEM_VIRT_MASK = ((1U << MEM_VIRT_WIDTH) - 1) << MEM_VIRT_BIT;

inline int32_t GetMallocMemType(unsigned long long flag)
{
    return (flag & 0b11110000000000) >> MEM_VIRT_BIT;
    //return flag & MEM_VIRT_MASK;
}

drvError_t halMemAlloc(void **pp, unsigned long long size, unsigned long long flag)
{
    drvError_t ret = halMemAllocInner(pp, size, flag);
    if (ret != DRV_ERROR_NONE) {
        return ret;
    }

    /*
    // 打印进程，线程号
    // 获取当前进程号
    pid_t pid = getpid();
    // 获取当前线程号
    std::thread::id tid = std::this_thread::get_id();
    // 获取父进程
    pid_t parent_pid = getppid();
    std::cout<<"Malloc当前线程："<< tid << " ,当前进程："<< pid << " ,当前父进程： " << parent_pid << std::endl;
    */

    // report to leaks here
    uint64_t addr = reinterpret_cast<uint64_t>(*pp);
    int32_t memType = GetMallocMemType(flag);
    MemOpSpace space = MemOpSpace::INVALID;
    switch (memType) {
        case MEM_SVM_VAL:
            space = MemOpSpace::SVM;
            break;
        case MEM_DEV_VAL:
            space = MemOpSpace::DEVICE;
            break;
        case MEM_HOST_VAL:
            space = MemOpSpace::HOST;
            break;
        case MEM_DVPP_VAL:
            space = MemOpSpace::DVPP;
            break;
        default:
            Utility::LogError("No matching memType for %d .", memType);   
    }
    if (!EventReport::Instance().ReportMalloc(addr, size, space, flag)) {
        Utility::LogError("Report FAILED");
    }

    return ret;
}

drvError_t halMemFree(void *pp)
{
    // report to leaks here
    uint64_t addr = reinterpret_cast<uint64_t>(pp);
    if (!EventReport::Instance().ReportFree(addr)) {
        Utility::LogError("Report FAILED");
    }

    drvError_t ret = halMemFreeInner(pp);
    if (ret != DRV_ERROR_NONE) {
        return ret;
    }

    /*
    // 打印进程，线程号
    // 获取当前进程号
    pid_t pid = getpid();
    // 获取当前线程号
    std::thread::id tid = std::this_thread::get_id();
    // 获取父进程
    pid_t parent_pid = getppid();
    std::cout<<"Free当前线程："<< tid << " ,当前进程："<< pid << " ,当前父进程： " << parent_pid << std::endl;
    */

    return ret;
}