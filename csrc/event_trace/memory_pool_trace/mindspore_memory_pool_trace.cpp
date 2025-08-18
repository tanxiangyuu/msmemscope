// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
#include "mindspore_memory_pool_trace.h"

#include <string>
#include "utils.h"
#include "event_report.h"
#include "call_stack.h"
#include "describe_trace.h"

namespace Leaks {
MindsporeMemoryPoolTrace::MindsporeMemoryPoolTrace()
{
    mindsporeDomain_ = new mstxDomainRegistration_st { };
}

MindsporeMemoryPoolTrace::~MindsporeMemoryPoolTrace()
{
    delete mindsporeDomain_;
    mindsporeDomain_ = nullptr;
}

mstxMemHeapHandle_t MindsporeMemoryPoolTrace::Allocate(mstxDomainHandle_t domain, mstxMemHeapDesc_t const *desc)
{
    if (domain == nullptr || desc == nullptr || domain != mindsporeDomain_) {
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

void MindsporeMemoryPoolTrace::Deallocate(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap)
{
    if (domain == nullptr || heap == nullptr || domain != mindsporeDomain_) {
        return;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    auto ptr = reinterpret_cast<void *>(heap);
    if (heapHandleMp_.count(ptr)) {
        mstxMemVirtualRangeDesc_t &rangeDesc = heapHandleMp_[ptr];
        if (memUsageMp_.count(rangeDesc.deviceId)) {
            memUsageMp_[rangeDesc.deviceId].totalReserved =
                Utility::GetSubResult(memUsageMp_[rangeDesc.deviceId].totalReserved, rangeDesc.size);
        }
    }
    return;
}

void MindsporeMemoryPoolTrace::Reallocate(mstxDomainHandle_t domain, mstxMemRegionsRegisterBatch_t const *desc)
{
    if (domain == nullptr || desc == nullptr || domain != mindsporeDomain_) {
        return;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    CallStackString stack;
    Utility::GetCallstack(stack);
    const mstxMemVirtualRangeDesc_t *rangeDescArray =
        reinterpret_cast<const mstxMemVirtualRangeDesc_t *>(desc->regionDescArray);

    for (size_t i = 0; i < desc->regionCount; i++) {
        uint32_t devId = rangeDescArray[i].deviceId;
        memUsageMp_[devId].dataType = 0;
        memUsageMp_[devId].deviceIndex = devId;
        memUsageMp_[devId].ptr = reinterpret_cast<int64_t>(rangeDescArray[i].ptr);
        memUsageMp_[devId].allocSize = rangeDescArray[i].size;
        memUsageMp_[devId].totalAllocated =
            Utility::GetAddResult(memUsageMp_[devId].totalAllocated, memUsageMp_[devId].allocSize);
        regionHandleMp_[rangeDescArray[i].ptr] = rangeDescArray[i];
        std::string owner = DescribeTrace::GetInstance().GetDescribe();
        TLVBlockType cStackType = stack.cStack.empty() ? TLVBlockType::SKIP : TLVBlockType::CALL_STACK_C;
        TLVBlockType pyStackType = stack.pyStack.empty() ? TLVBlockType::SKIP : TLVBlockType::CALL_STACK_PYTHON;
        RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<MemPoolRecord>(
            TLVBlockType::ADDR_OWNER, owner.c_str(), cStackType, stack.cStack, pyStackType, stack.pyStack);
        MemPoolRecord* record = buffer.Cast<MemPoolRecord>();
        record->type = RecordType::MINDSPORE_NPU_RECORD;
        record->memoryUsage = memUsageMp_[devId];
        if (!EventReport::Instance(CommType::SOCKET).ReportMemPoolRecord(buffer)) {
            CLIENT_ERROR_LOG("Report Mindspore Data Failed");
        }
    }
}

void MindsporeMemoryPoolTrace::Release(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const *desc)
{
    if (domain == nullptr || desc == nullptr || domain != mindsporeDomain_) {
        return;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    CallStackString stack;
    Utility::GetCallstack(stack);
    for (size_t i = 0; i < desc->refCount; i++) {
        if (!regionHandleMp_.count(desc->refArray[i].pointer)) {
            continue;
        }
        mstxMemVirtualRangeDesc_t rangeDesc = regionHandleMp_[desc->refArray[i].pointer];
        memUsageMp_[rangeDesc.deviceId].dataType = 1;
        memUsageMp_[rangeDesc.deviceId].deviceIndex = rangeDesc.deviceId;
        memUsageMp_[rangeDesc.deviceId].ptr = reinterpret_cast<int64_t>(rangeDesc.ptr);
        memUsageMp_[rangeDesc.deviceId].totalAllocated =
            Utility::GetSubResult(memUsageMp_[rangeDesc.deviceId].totalAllocated, rangeDesc.size);
        memUsageMp_[rangeDesc.deviceId].allocSize = rangeDesc.size;
        std::string owner = " ";
        TLVBlockType cStackType = stack.cStack.empty() ? TLVBlockType::SKIP : TLVBlockType::CALL_STACK_C;
        TLVBlockType pyStackType = stack.pyStack.empty() ? TLVBlockType::SKIP : TLVBlockType::CALL_STACK_PYTHON;
        RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<MemPoolRecord>(
            TLVBlockType::ADDR_OWNER, owner.c_str(), cStackType, stack.cStack, pyStackType, stack.pyStack);
        MemPoolRecord* record = buffer.Cast<MemPoolRecord>();
        record->type = RecordType::MINDSPORE_NPU_RECORD;
        record->memoryUsage = memUsageMp_[rangeDesc.deviceId];
        if (!EventReport::Instance(CommType::SOCKET).ReportMemPoolRecord(buffer)) {
            CLIENT_ERROR_LOG("Report Mindspore Data Failed");
        }
    }
}

mstxDomainHandle_t MindsporeMemoryPoolTrace::CreateDomain(const std::string &domainName)
{
    if (domainName == "mindsporeMemPool") {
        return mindsporeDomain_;
    }
    return nullptr;
}
}