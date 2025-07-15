// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef PTA_MEMORY_POOL_TRACE_H
#define PTA_MEMORY_POOL_TRACE_H

#include <mutex>
#include <unordered_map>
#include "memory_pool_trace_base.h"
#include "record_info.h"

namespace Leaks {

class PTAMemoryPoolTrace : public MemoryPoolTraceBase {
public:
    PTAMemoryPoolTrace(const PTAMemoryPoolTrace&) = delete;
    PTAMemoryPoolTrace& operator=(const PTAMemoryPoolTrace&) = delete;

    static PTAMemoryPoolTrace& GetInstance()
    {
        static PTAMemoryPoolTrace instance;
        return instance;
    }

    mstxMemHeapHandle_t Allocate(mstxDomainHandle_t domain, mstxMemHeapDesc_t const *desc) override;

    void Deallocate(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap) override;

    void Reallocate(mstxDomainHandle_t domain, mstxMemRegionsRegisterBatch_t const *desc) override;

    void Release(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const *desc) override;

    mstxDomainHandle_t CreateDomain(const std::string &domainName) override;

private:
    PTAMemoryPoolTrace();
    ~PTAMemoryPoolTrace() override;

    mstxDomainHandle_t ptaDomain_ { nullptr };
    std::unordered_map<uint32_t, MemoryUsage> memUsageMp_; // 单进程可能涉及多张卡
    std::unordered_map<const void*, mstxMemVirtualRangeDesc_t> regionHandleMp_;
    std::unordered_map<const void*, mstxMemVirtualRangeDesc_t> heapHandleMp_;

    std::mutex mutex_;
};
}
#endif
