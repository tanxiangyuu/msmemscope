// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "hal_hooks.h"
#include <string>
#include <dlfcn.h>
#include "event_report.h"
#include "log.h"

using namespace Leaks;

constexpr uint64_t MEM_VIRT_BIT = 10;
constexpr uint64_t MEM_VIRT_WIDTH = 4;
constexpr uint64_t MEM_DEV_VAL = 0x1;
constexpr uint64_t MEM_HOST_VAL = 0x2;
constexpr uint64_t MEM_HOST = MEM_HOST_VAL << MEM_VIRT_BIT;
constexpr uint64_t MEM_DEV = MEM_DEV_VAL << MEM_VIRT_BIT;
constexpr uint64_t MEM_VIRT_MASK = ((1U << MEM_VIRT_WIDTH) - 1) << MEM_VIRT_BIT;

inline int32_t GetMallocModuleId(unsigned long long flag)
{
    return flag & MEM_VIRT_MASK;
}

drvError_t halMemAlloc(void **pp, unsigned long long size, unsigned long long flag)
{
    drvError_t ret = halMemAllocInner(pp, size, flag);
    if (ret != DRV_ERROR_NONE) {
        return ret;
    }

    // report to leaks here
    uint64_t addr = reinterpret_cast<uint64_t>(*pp);
    int32_t moduleId = GetMallocModuleId(flag);
    MemOpSpace space = (moduleId == MEM_HOST ? MemOpSpace::HOST : MemOpSpace::DEVICE);
    if (!EventReport::Instance().ReportMalloc(addr, size, space)) {
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

    return ret;
}