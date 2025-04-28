// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef MEMORY_POOL_TRACE_BASE_H
#define MEMORY_POOL_TRACE_BASE_H

#include <string>
#include "mstx_hooks/mstx_info.h"

namespace Leaks {

// 通过MSTX监控device内存池抽象基类，不同内存池完成各自的泛化处理
class MemoryPoolTraceBase {
public:
    virtual ~MemoryPoolTraceBase() = default;

    // 内存池分配
    virtual mstxMemHeapHandle_t Allocate(mstxDomainHandle_t domain, mstxMemHeapDesc_t const *desc) = 0;

    // 内存池释放
    virtual void Deallocate(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap) = 0;

    // 二次分配
    virtual void Reallocate(mstxDomainHandle_t domain, mstxMemRegionsRegisterBatch_t const *desc) = 0;

    // 二次释放
    virtual void Release(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const *desc) = 0;

    // 维护domain
    virtual mstxDomainHandle_t CreateDomain(std::string domainName) = 0;
    
protected:
    MemoryPoolTraceBase() = default;
};
}
#endif