// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
#include <string>
#include "utils.h"
#include "event_report.h"
#include "call_stack.h"
#include "atb_memory_pool_trace.h"

namespace Leaks {
ATBMemoryPoolTrace::ATBMemoryPoolTrace()
{
    atbDomain_ = new mstxDomainRegistration_st { };
}

ATBMemoryPoolTrace::~ATBMemoryPoolTrace()
{
    delete atbDomain_;
    atbDomain_ = nullptr;
}

mstxMemHeapHandle_t ATBMemoryPoolTrace::Allocate(mstxDomainHandle_t domain, mstxMemHeapDesc_t const *desc)
{
    if (domain == nullptr || desc == nullptr || domain != atbDomain_) {
        return nullptr;
    }
    std::lock_guard<std::mutex> guard(mutex_);

    const mstxMemVirtualRangeDesc_t *rangeDesc =
        reinterpret_cast<const mstxMemVirtualRangeDesc_t *>(desc->typeSpecificDesc);
    int64_t memPoolSize = rangeDesc->size;
    
    if (memUsageMp_.count(rangeDesc->deviceId)) {
        memUsageMp_[rangeDesc->deviceId].totalReserved =
            Utility::GetAddResult(memUsageMp_[rangeDesc->deviceId].totalReserved, memPoolSize);
    } else {
        auto memoryUsage = MemoryUsage { };
        memoryUsage.totalReserved = memPoolSize;
        memUsageMp_.insert({rangeDesc->deviceId, memoryUsage});
    }

    heapHandleMp_[rangeDesc->ptr] = *rangeDesc;

    return const_cast<mstxMemHeapHandle_t>(reinterpret_cast<const mstxMemHeap_t*>(rangeDesc->ptr));
}

void ATBMemoryPoolTrace::Deallocate(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap)
{
    if (domain == nullptr || heap == nullptr || domain != atbDomain_) {
        return;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    auto handle = reinterpret_cast<void *>(heap);
    if (heapHandleMp_.count(handle)) {
        mstxMemVirtualRangeDesc_t &rangeDesc = heapHandleMp_[handle];
        if (memUsageMp_.count(rangeDesc.deviceId)) {
            memUsageMp_[rangeDesc.deviceId].totalReserved =
                Utility::GetSubResult(memUsageMp_[rangeDesc.deviceId].totalReserved, rangeDesc.size);
        }
        heapHandleMp_.erase(handle);
    }
    return;
}

void ATBMemoryPoolTrace::Reallocate(mstxDomainHandle_t domain, mstxMemRegionsRegisterBatch_t const *desc)
{
    if (domain == nullptr || desc == nullptr || domain != atbDomain_) {
        return;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    auto config = EventReport::Instance(CommType::SOCKET).GetConfig();
    std::string cStack;
    std::string pyStack;
    if (config.enableCStack) {
        Utility::GetCCallstack(config.cStackDepth, cStack, SKIP_DEPTH);
    }
    if (config.enablePyStack) {
        Utility::GetPythonCallstack(config.pyStackDepth, pyStack);
    }
    CallStackString stack{pyStack, cStack};
    const mstxMemVirtualRangeDesc_t *rangeDescArray =
        reinterpret_cast<const mstxMemVirtualRangeDesc_t *>(desc->regionDescArray);

    for (size_t i = 0; i < desc->regionCount; i++) {
        uint32_t devId = rangeDescArray[i].deviceId;
        
        if (!memUsageMp_.count(devId)) {
            continue;
        }
        memUsageMp_[devId].dataType = 0;
        memUsageMp_[devId].deviceIndex = devId;
        memUsageMp_[devId].ptr = reinterpret_cast<int64_t>(rangeDescArray[i].ptr);
        memUsageMp_[devId].allocSize = rangeDescArray[i].size;
        memUsageMp_[devId].totalAllocated =
            Utility::GetAddResult(memUsageMp_[devId].totalAllocated, memUsageMp_[devId].allocSize);
        auto handle = new mstxMemRegion_t {};
        desc->regionHandleArrayOut[i] = handle;
        regionHandleMp_[handle] = rangeDescArray[i];

        AtbMemPoolRecord record;
        record.memoryUsage = memUsageMp_[devId];
        record.pid = Utility::GetPid();
        record.tid = Utility::GetTid();
        if (!EventReport::Instance(CommType::SOCKET).ReportATBMemPoolRecord(record, stack)) {
            CLIENT_ERROR_LOG("Report ATB Data Failed");
        }
    }
}

void ATBMemoryPoolTrace::Release(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const *desc)
{
    if (domain == nullptr || desc == nullptr || domain != atbDomain_) {
        return;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    auto config = EventReport::Instance(CommType::SOCKET).GetConfig();
    std::string cStack;
    std::string pyStack;
    if (config.enableCStack) {
        Utility::GetCCallstack(config.cStackDepth, cStack, SKIP_DEPTH);
    }
    if (config.enablePyStack) {
        Utility::GetPythonCallstack(config.pyStackDepth, pyStack);
    }
    CallStackString stack{pyStack, cStack};
    for (size_t i = 0; i < desc->refCount; i++) {
        auto handle = desc->refArray[i].handle;
        auto iter = regionHandleMp_.find(handle);
        if (iter == regionHandleMp_.end()) {
            continue;
        }
        AtbMemPoolRecord record;
        mstxMemVirtualRangeDesc_t rangeDesc = iter->second;
        memUsageMp_[rangeDesc.deviceId].dataType = 1;
        memUsageMp_[rangeDesc.deviceId].deviceIndex = rangeDesc.deviceId;
        memUsageMp_[rangeDesc.deviceId].ptr = reinterpret_cast<int64_t>(rangeDesc.ptr);
        memUsageMp_[rangeDesc.deviceId].totalAllocated =
            Utility::GetSubResult(memUsageMp_[rangeDesc.deviceId].totalAllocated, rangeDesc.size);
        memUsageMp_[rangeDesc.deviceId].allocSize = -rangeDesc.size;
        record.memoryUsage = memUsageMp_[rangeDesc.deviceId];
        record.pid = Utility::GetPid();
        record.tid = Utility::GetTid();

        regionHandleMp_.erase(handle);
        delete handle;
        handle = nullptr;
        if (!EventReport::Instance(CommType::SOCKET).ReportATBMemPoolRecord(record, stack)) {
            CLIENT_ERROR_LOG("Report ATB Data Failed");
        }
    }
}

mstxDomainHandle_t ATBMemoryPoolTrace::CreateDomain(const std::string &domainName)
{
    if (domainName == "atb") {
        return atbDomain_;
    }
    return nullptr;
}
}