/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

#include "hal_hooks.h"
#include <string>
#include <dlfcn.h>
#include <iostream>
#include "call_stack.h"
#include "log.h"
#include "trace_manager/event_trace_manager.h"

using namespace MemScope;

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

    // report to memscope here
    uintptr_t addr = reinterpret_cast<uintptr_t>(*pp);
    if (!EventReport::Instance(MemScopeCommType::SHARED_MEMORY)
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
    if (!EventReport::Instance(MemScopeCommType::SHARED_MEMORY).ReportFree(reinterpret_cast<uint64_t>(addr), stack)) {
        LOG_ERROR("halMemFree report failed");
    }

    return ret;
}