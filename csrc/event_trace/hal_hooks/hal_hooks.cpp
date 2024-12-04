// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "hal_hooks.h"
#include <string>
#include <dlfcn.h>


#include "log.h"

using namespace Leaks;

MemOpSpace GetMemOpSpace(unsigned long long flag)
{
    // bit10~13: virt mem type(svm\dev\host\dvpp)
    int32_t memType = (flag & 0b11110000000000) >> MEM_VIRT_BIT;
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
    return space;
}

drvError_t halMemAlloc(void **pp, unsigned long long size, unsigned long long flag)
{
    drvError_t ret = halMemAllocInner(pp, size, flag);
    if (ret != DRV_ERROR_NONE) {
        return ret;
    }

    // report to leaks here
    uint64_t addr = reinterpret_cast<uint64_t>(*pp);
    MemOpSpace space = GetMemOpSpace(flag);
    if (!EventReport::Instance(CommType::SOCKET).ReportMalloc(addr, size, space, flag)) {
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