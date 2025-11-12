// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "hal_hooks.h"
#include <string>
#include <dlfcn.h>
#include <iostream>
#include "call_stack.h"
#include "log.h"
#include "trace_manager/event_trace_manager.h"

using namespace Leaks;

drvError_t halMemAlloc(void **pp, unsigned long long size, unsigned long long flag)
{
    drvError_t ret = halMemAllocInner(pp, size, flag);
    if (ret != DRV_ERROR_NONE) {
        return ret;
    }
    if (!EventTraceManager::Instance().IsTracingEnabled() ||
        !EventTraceManager::Instance().ShouldTraceType(RecordType::MEMORY_RECORD)) {
        return ret;
    }

    CallStackString stack;
    Utility::GetCallstack(stack);

    // report to leaks here
    uintptr_t addr = reinterpret_cast<uintptr_t>(*pp);
    if (!EventReport::Instance(LeaksCommType::SHARED_MEMORY)
             .ReportMalloc(reinterpret_cast<uint64_t>(addr), size, flag, stack)) {
        LOG_ERROR("halMemAlloc report failed");
    }

    return ret;
}

drvError_t halMemFree(void *pp)
{
    drvError_t ret = halMemFreeInner(pp);
    if (ret != DRV_ERROR_NONE) {
        return ret;
    }

    if (!EventTraceManager::Instance().IsTracingEnabled() ||
        !EventTraceManager::Instance().ShouldTraceType(RecordType::MEMORY_RECORD)) {
        return ret;
    }

    CallStackString stack;
    Utility::GetCallstack(stack);
    
    uintptr_t addr = reinterpret_cast<uintptr_t>(pp);
    if (!EventReport::Instance(LeaksCommType::SHARED_MEMORY).ReportFree(reinterpret_cast<uint64_t>(addr), stack)) {
        LOG_ERROR("halMemFree report failed");
    }

    return ret;
}