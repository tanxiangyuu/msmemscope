// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "hal_hooks.h"
#include <string>
#include <dlfcn.h>
#include <iostream>

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
    if (!EventReport::Instance(CommType::SOCKET).ReportMalloc(addr, size, flag)) {
        std::cout << "halMemAlloc report FAILED" << std::endl;
    }

    return ret;
}

drvError_t halMemFree(void *pp)
{
    // report to leaks here
    uint64_t addr = reinterpret_cast<uint64_t>(pp);
    
    if (!EventReport::Instance(CommType::SOCKET).ReportFree(addr)) {
        std::cout << "halMemFree report FAILED" << std::endl;
    }

    drvError_t ret = halMemFreeInner(pp);
    if (ret != DRV_ERROR_NONE) {
        return ret;
    }

    return ret;
}