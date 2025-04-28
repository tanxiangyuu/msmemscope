// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "memory_pool_trace_manager.h"

namespace Leaks {

mstxMemHeapHandle_t MemoryPoolTraceManager::Allocate(mstxDomainHandle_t domain, mstxMemHeapDesc_t const *desc)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto tracer = GetMemoryPoolTracer(domain);
    if (tracer) {
        return tracer->Allocate(domain, desc);
    }
    return nullptr;
}

void MemoryPoolTraceManager::Deallocate(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto tracer = GetMemoryPoolTracer(domain);
    if (tracer) {
        tracer->Deallocate(domain, heap);
    }
}

void MemoryPoolTraceManager::Reallocate(mstxDomainHandle_t domain, mstxMemRegionsRegisterBatch_t const *desc)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto tracer = GetMemoryPoolTracer(domain);
    if (tracer) {
        tracer->Reallocate(domain, desc);
    }
}

void MemoryPoolTraceManager::Release(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const *desc)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto tracer = GetMemoryPoolTracer(domain);
    if (tracer) {
        tracer->Release(domain, desc);
    }
}

mstxDomainHandle_t MemoryPoolTraceManager::CreateDomain(const std::string& domainName)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto tracer = GetMemoryPoolTracer(domainName);
    if (tracer) {
        return tracer->CreateDomain(domainName);
    }
    return nullptr;
}

bool MemoryPoolTraceManager::RegisterMemoryPoolTracer(const std::string& domainName, MemoryPoolTraceBase* tracer)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (tracers_.find(domainName) != tracers_.end()) {
        return false;
    }
    tracers_[domainName] = tracer;

    return true;
}

MemoryPoolTraceBase* MemoryPoolTraceManager::GetMemoryPoolTracer(const std::string& domainName)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tracers_.find(domainName);
    if (it != tracers_.end()) {
        return it->second;
    }
    return nullptr;
}

MemoryPoolTraceBase* MemoryPoolTraceManager::GetMemoryPoolTracer(const mstxDomainHandle_t& domainHandle)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &pair : domainMp_) {
        if (pair.second == domainHandle) {
            return GetMemoryPoolTracer(pair.first);
        }
    }
    return nullptr;
}
}