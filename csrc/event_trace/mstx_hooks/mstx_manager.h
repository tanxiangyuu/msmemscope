// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#ifndef MSLEAKS_MSTX_MANAGER_H
#define MSLEAKS_MSTX_MANAGER_H

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <iostream>
#include "record_info.h"
#include "mstx_info.h"

namespace Leaks {
/*
* mstx_inject仅仅做接口的转发调用，实际的上报信息组装由Manager来完成，避免接口变动时改动过大
*/
class MstxManager {
public:
    static MstxManager& GetInstance()
    {
        static MstxManager instance;
        return instance;
    }
    MstxManager(const MstxManager&) = delete;
    MstxManager& operator=(const MstxManager&) = delete;

    void ReportMarkA(const char* msg, int32_t stream);
    uint64_t ReportRangeStart(const char* msg, int32_t stream);
    void ReportRangeEnd(uint64_t id);
    mstxDomainHandle_t ReportDomainCreateA(char const *domainName);
    mstxMemHeapHandle_t ReportHeapRegister(mstxDomainHandle_t domain, mstxMemHeapDesc_t const *desc);
    void ReportHeapUnregister(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap);
    void ReportRegionsRegister(mstxDomainHandle_t domain, mstxMemRegionsRegisterBatch_t const *desc);
    void ReportRegionsUnregister(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const *desc);

private:
    MstxManager()
    {
        msleaksDomain_ = new mstxDomainRegistration_st;
        rangeId_ = 1;
    }
    ~MstxManager()
    {
        delete msleaksDomain_;
        msleaksDomain_ = nullptr;
    }
    uint64_t GetRangeId();

private:
    std::mutex mutex_;
    mstxDomainHandle_t msleaksDomain_{nullptr};
    std::unordered_map<int, MemoryUsage> memUsageMp_;
    std::unordered_map<const void*, mstxMemVirtualRangeDesc_t> regionHandleMp_;
    std::unordered_map<const void*, mstxMemVirtualRangeDesc_t> heapHandleMp_;
    std::atomic<uint64_t> rangeId_;
    const int onlyMarkId_ = 0;
};

}

#endif