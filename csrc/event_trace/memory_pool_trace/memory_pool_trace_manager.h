// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef MEMORY_POOL_TRACE_MANAGER_H
#define MEMORY_POOL_TRACE_MANAGER_H

#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>
#include "memory_pool_trace_base.h"

namespace Leaks {

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