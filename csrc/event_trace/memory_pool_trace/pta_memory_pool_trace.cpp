// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
#include <string>
#include "utils.h"
#include "event_report.h"
#include "call_stack.h"
#include "pta_memory_pool_trace.h"

namespace Leaks {
PTAMemoryPoolTrace::PTAMemoryPoolTrace()
{
    ptaDomain_ = new mstxDomainRegistration_st { };
}

PTAMemoryPoolTrace::~PTAMemoryPoolTrace()
{
    delete ptaDomain_;
    ptaDomain_ = nullptr;
}

mstxMemHeapHandle_t PTAMemoryPoolTrace::Allocate(mstxDomainHandle_t domain, mstxMemHeapDesc_t const *desc)
{
    if (domain == nullptr || desc == nullptr || domain != ptaDomain_) {
        return nullptr;
    }
    std::lock_guard<std::mutex> guard(mutex_);

    const mstxMemVirtualRangeDesc_t *rangeDesc =
        reinterpret_cast<const mstxMemVirtualRangeDesc_t *>(desc->typeSpecificDesc);
    int64_t memSize = rangeDesc->size;
    int devId = rangeDesc->deviceId;
    memUsageMp_[devId].totalReserved = memSize;
    heapHandleMp_[rangeDesc->ptr] = *rangeDesc;

    return nullptr;
}

void PTAMemoryPoolTrace::Deallocate(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap)
{
    if (domain == nullptr || heap == nullptr || domain != ptaDomain_) {
        return;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    void *ptr = reinterpret_cast<void *>(heap);
    if (heapHandleMp_.count(ptr)) {
        auto desc = heapHandleMp_[ptr];
        memUsageMp_[desc.deviceId].totalReserved =
            Utility::GetSubResult(memUsageMp_[desc.deviceId].totalReserved, desc.size);
    }
}

void PTAMemoryPoolTrace::Reallocate(mstxDomainHandle_t domain, mstxMemRegionsRegisterBatch_t const *desc)
{
    if (domain == nullptr || desc == nullptr || domain != ptaDomain_) {
        return;
    }
    std::lock_guard<std::mutex> guard(mutex_);

    const mstxMemVirtualRangeDesc_t *rangeDescArray =
        reinterpret_cast<const mstxMemVirtualRangeDesc_t *>(desc->regionDescArray);
    auto config = EventReport::Instance(CommType::SOCKET).GetConfig();
    std::string cStack;
    std::string pyStack;
    if (config.enableCStack) {
        Utility::GetCCallstack(config.cStackDepth, cStack, SKIP_DEPTH);
    }
    if (config.enablePyStack) {
        Utility::GetPythonCallstack(config.pyStackDepth, pyStack);
    }
    CallStackString stack{cStack, pyStack};

    for (size_t i = 0; i < desc->regionCount; i++) {
        uint32_t devId = rangeDescArray[i].deviceId;
        memUsageMp_[devId].dataType = 0;
        memUsageMp_[devId].deviceIndex = devId;
        memUsageMp_[devId].ptr = reinterpret_cast<int64_t>(rangeDescArray[i].ptr);
        memUsageMp_[devId].allocSize = rangeDescArray[i].size;
        memUsageMp_[devId].totalAllocated =
            Utility::GetAddResult(memUsageMp_[devId].totalAllocated, memUsageMp_[devId].allocSize);
        regionHandleMp_[rangeDescArray[i].ptr] = rangeDescArray[i];

        MemPoolRecord memPoolRecord;
        memPoolRecord.type = RecordType::TORCH_NPU_RECORD;
        memPoolRecord.memoryUsage = memUsageMp_[devId];
        memPoolRecord.pid = Utility::GetPid();
        memPoolRecord.tid = Utility::GetTid();
        if (!EventReport::Instance(CommType::SOCKET).ReportMemPoolRecord(memPoolRecord, stack)) {
            CLIENT_ERROR_LOG("Report PTA Data Failed");
        }
    }
}

void PTAMemoryPoolTrace::Release(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const *desc)
{
    if (domain == nullptr || desc == nullptr || domain != ptaDomain_) {
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
    CallStackString stack{cStack, pyStack};
    for (size_t i = 0; i < desc->refCount; i++) {
        if (!regionHandleMp_.count(desc->refArray[i].pointer)) {
            continue;
        }
        MemPoolRecord memPoolRecord;
        mstxMemVirtualRangeDesc_t rangeDesc = regionHandleMp_[desc->refArray[i].pointer];
        memUsageMp_[rangeDesc.deviceId].dataType = 1;
        memUsageMp_[rangeDesc.deviceId].deviceIndex = rangeDesc.deviceId;
        memUsageMp_[rangeDesc.deviceId].ptr = reinterpret_cast<int64_t>(rangeDesc.ptr);
        memUsageMp_[rangeDesc.deviceId].totalAllocated =
            Utility::GetSubResult(memUsageMp_[rangeDesc.deviceId].totalAllocated, rangeDesc.size);
        memUsageMp_[rangeDesc.deviceId].allocSize = rangeDesc.size;
        memPoolRecord.type = RecordType::TORCH_NPU_RECORD;
        memPoolRecord.memoryUsage = memUsageMp_[rangeDesc.deviceId];
        memPoolRecord.pid = Utility::GetPid();
        memPoolRecord.tid = Utility::GetTid();
        if (!EventReport::Instance(CommType::SOCKET).ReportMemPoolRecord(memPoolRecord, stack)) {
            CLIENT_ERROR_LOG("Report PTA Data Failed");
        }
    }
}

mstxDomainHandle_t PTAMemoryPoolTrace::CreateDomain(const std::string &domainName)
{
    if (domainName == "msleaks") {
        return ptaDomain_;
    }
    return nullptr;
}
}