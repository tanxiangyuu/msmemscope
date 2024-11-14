// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "hal_hooks.h"
#include <string>
#include <dlfcn.h>
#include "event_report.h"
#include "log.h"

using namespace Leaks;

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
    if (!EventReport::Instance(CommType::SOCKET).ReportMalloc(addr, size, space)) {
        Utility::LogError("%s report FAILED", __func__);
    }

    return ret;
}

drvError_t halMemFree(void *pp)
{
    // report to leaks here
    uint64_t addr = reinterpret_cast<uint64_t>(pp);
    if (!EventReport::Instance(CommType::SOCKET).ReportFree(addr)) {
        Utility::LogError("%s report FAILED", __func__);
    }

    drvError_t ret = halMemFreeInner(pp);
    if (ret != DRV_ERROR_NONE) {
        return ret;
    }

    return ret;
}