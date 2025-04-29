// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "memory_pool_trace_manager.h"

namespace Leaks {

mstxMemHeapHandle_t MemoryPoolTraceManager::Allocate(mstxDomainHandle_t domain, mstxMemHeapDesc_t const *desc)
{
    auto tracer = GetMemoryPoolTracer(domain);
    if (tracer) {
        return tracer->Allocate(domain, desc);
    }
    return nullptr;
}

void MemoryPoolTraceManager::Deallocate(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap)
{
    auto tracer = GetMemoryPoolTracer(domain);
    if (tracer) {
        tracer->Deallocate(domain, heap);
    }
    return;
}

void MemoryPoolTraceManager::Reallocate(mstxDomainHandle_t domain, mstxMemRegionsRegisterBatch_t const *desc)
{
    auto tracer = GetMemoryPoolTracer(domain);
    if (tracer) {
        tracer->Reallocate(domain, desc);
    }
    return;
}

void MemoryPoolTraceManager::Release(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const *desc)
{
    auto tracer = GetMemoryPoolTracer(domain);
    if (tracer) {
        tracer->Release(domain, desc);
    }
    return;
}

mstxDomainHandle_t MemoryPoolTraceManager::CreateDomain(const std::string& domainName)
{
    auto tracer = GetMemoryPoolTracer(domainName);
    if (tracer) {
        auto domainHandle = tracer->CreateDomain(domainName);
        if (domainMp_.find(domainName) == domainMp_.end()) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                domainMp_[domainName] = domainHandle;
            }
        }
        return domainHandle;
    }
    return nullptr;
}

bool MemoryPoolTraceManager::RegisterMemoryPoolTracer(const std::string& domainName, MemoryPoolTraceBase* tracer)
{
    if (tracers_.find(domainName) != tracers_.end()) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tracers_[domainName] = tracer;
    }

    return true;
}

MemoryPoolTraceBase* MemoryPoolTraceManager::GetMemoryPoolTracer(const std::string& domainName)
{
    auto it = tracers_.find(domainName);
    if (it != tracers_.end()) {
        return it->second;
    }
    return nullptr;
}

MemoryPoolTraceBase* MemoryPoolTraceManager::GetMemoryPoolTracer(const mstxDomainHandle_t& domainHandle)
{
    for (auto &pair : domainMp_) {
        if (pair.second == domainHandle) {
            return GetMemoryPoolTracer(pair.first);
        }
    }
    return nullptr;
}
}