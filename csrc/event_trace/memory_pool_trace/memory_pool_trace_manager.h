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

#ifndef MEMORY_POOL_TRACE_MANAGER_H
#define MEMORY_POOL_TRACE_MANAGER_H

#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>
#include "memory_pool_trace_base.h"

namespace MemScope {

class MemoryPoolTraceManager {
public:
    MemoryPoolTraceManager(const MemoryPoolTraceManager&) = delete;
    MemoryPoolTraceManager& operator=(const MemoryPoolTraceManager&) = delete;

    static MemoryPoolTraceManager& GetInstance()
    {
        static MemoryPoolTraceManager instance;
        return instance;
    }

    mstxMemHeapHandle_t Allocate(mstxDomainHandle_t domain, mstxMemHeapDesc_t const *desc);
    void Deallocate(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap);
    void Reallocate(mstxDomainHandle_t domain, mstxMemRegionsRegisterBatch_t const *desc);
    void Release(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const *desc);

    mstxDomainHandle_t CreateDomain(const std::string& domainName);

    bool RegisterMemoryPoolTracer(const std::string& domainName, MemoryPoolTraceBase* tracer);

    MemoryPoolTraceBase* GetMemoryPoolTracer(const std::string& domainName);
    MemoryPoolTraceBase* GetMemoryPoolTracer(const mstxDomainHandle_t& domainHandle);
private:
    MemoryPoolTraceManager() = default;
    ~MemoryPoolTraceManager() = default;

    std::unordered_map<std::string, MemoryPoolTraceBase*> tracers_;
    std::unordered_map<std::string, mstxDomainHandle_t> domainMp_;
    std::mutex mutex_;
};

}
#endif