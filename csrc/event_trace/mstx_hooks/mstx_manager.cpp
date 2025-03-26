// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "mstx_manager.h"
#include <cstring>
#include <iostream>
#include "securec.h"
#include "event_report.h"
#include "record_info.h"
#include "log.h"

namespace Leaks {
// 组装普通打点信息
void MstxManager::ReportMarkA(const char* msg, int32_t streamId)
{
    MstxRecord record;
    record.markType = MarkType::MARK_A;
    record.rangeId = onlyMarkId_;
    record.streamId = streamId;

    if (strncpy_s(record.markMessage, sizeof(record.markMessage), msg, sizeof(record.markMessage) - 1) != EOK) {
        CLIENT_ERROR_LOG("strncpy_s FAILED");
    }
    record.markMessage[sizeof(record.markMessage) - 1] = '\0';
    if (!EventReport::Instance(CommType::SOCKET).ReportMark(record)) {
        CLIENT_ERROR_LOG("Report Mark FAILED");
    }
}

// 组装Range开始打点信息
uint64_t MstxManager::ReportRangeStart(const char* msg, int32_t streamId)
{
    MstxRecord record;
    record.markType = MarkType::RANGE_START_A;
    record.streamId = streamId;
    if (strncpy_s(record.markMessage, sizeof(record.markMessage), msg, sizeof(record.markMessage) - 1) != EOK) {
        CLIENT_ERROR_LOG("strncpy_s FAILED");
    }
    record.markMessage[sizeof(record.markMessage) - 1] = '\0';
    record.rangeId = GetRangeId();
    if (!EventReport::Instance(CommType::SOCKET).ReportMark(record)) {
        CLIENT_ERROR_LOG("Report Mark FAILED");
    }
    return record.rangeId;
}

// 组装Range结束打点信息
void MstxManager::ReportRangeEnd(uint64_t id)
{
    MstxRecord record;
    record.markType = MarkType::RANGE_END;
    record.rangeId = id;
    record.streamId = -1;
    std::string msg = "Range end from id " + std::to_string(id);
    if (strncpy_s(record.markMessage, sizeof(record.markMessage), msg.c_str(), sizeof(record.markMessage) - 1) != EOK) {
        CLIENT_ERROR_LOG("strncpy_s FAILED");
    }
    record.markMessage[sizeof(record.markMessage) - 1] = '\0';
    if (!EventReport::Instance(CommType::SOCKET).ReportMark(record)) {
        CLIENT_ERROR_LOG("Report Mark FAILED");
    }
}

uint64_t MstxManager::GetRangeId()
{
    return rangeId_++;
}

mstxDomainHandle_t MstxManager::ReportDomainCreateA(char const *domainName)
{
    if (domainName == nullptr || std::string(domainName) != "msleaks") {
        return nullptr;
    }
    return msleaksDomain_;
}

mstxMemHeapHandle_t MstxManager::ReportHeapRegister(mstxDomainHandle_t domain, mstxMemHeapDesc_t const *desc)
{
    if (domain == nullptr || desc == nullptr || domain != msleaksDomain_) {
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

void MstxManager::ReportHeapUnregister(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap)
{
    if (domain == nullptr || heap == nullptr || domain != msleaksDomain_) {
        return;
    }
    void *ptr = reinterpret_cast<void *>(heap);
    if (heapHandleMp_.count(ptr)) {
        auto desc = heapHandleMp_[ptr];
        memUsageMp_[desc.deviceId].totalReserved =
            Utility::GetSubResult(memUsageMp_[desc.deviceId].totalReserved, desc.size);
    }
}

void MstxManager::ReportRegionsRegister(mstxDomainHandle_t domain, mstxMemRegionsRegisterBatch_t const *desc)
{
    if (domain == nullptr || desc == nullptr || domain != msleaksDomain_) {
        return;
    }
    std::lock_guard<std::mutex> guard(mutex_);

    const mstxMemVirtualRangeDesc_t *rangeDescArray =
        reinterpret_cast<const mstxMemVirtualRangeDesc_t *>(desc->regionDescArray);

    for (size_t i = 0; i < desc->regionCount; i++) {
        TorchNpuRecord torchNpuRecord;
        int devId = rangeDescArray[i].deviceId;
        memUsageMp_[devId].dataType = 0;
        memUsageMp_[devId].deviceIndex = devId;
        memUsageMp_[devId].ptr = reinterpret_cast<int64_t>(rangeDescArray[i].ptr);
        memUsageMp_[devId].allocSize = rangeDescArray[i].size;
        memUsageMp_[devId].totalAllocated =
            Utility::GetAddResult(memUsageMp_[devId].totalAllocated, memUsageMp_[devId].allocSize);
        regionHandleMp_[rangeDescArray[i].ptr] = rangeDescArray[i];
        torchNpuRecord.memoryUsage = memUsageMp_[devId];
        torchNpuRecord.pid = Utility::GetPid();
        torchNpuRecord.tid = Utility::GetTid();
        if (!EventReport::Instance(CommType::SOCKET).ReportTorchNpu(torchNpuRecord)) {
            CLIENT_ERROR_LOG("Report Npu Data Failed");
        }
    }
}

void MstxManager::ReportRegionsUnregister(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const *desc)
{
    if (domain == nullptr || desc == nullptr || domain != msleaksDomain_) {
        return;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    for (size_t i = 0; i < desc->refCount; i++) {
        if (!regionHandleMp_.count(desc->refArray[i].pointer)) {
            continue;
        }
        TorchNpuRecord torchNpuRecord;
        mstxMemVirtualRangeDesc_t rangeDesc = regionHandleMp_[desc->refArray[i].pointer];
        memUsageMp_[rangeDesc.deviceId].dataType = 1;
        memUsageMp_[rangeDesc.deviceId].deviceIndex = rangeDesc.deviceId;
        memUsageMp_[rangeDesc.deviceId].ptr = reinterpret_cast<int64_t>(rangeDesc.ptr);
        memUsageMp_[rangeDesc.deviceId].totalAllocated =
            Utility::GetSubResult(memUsageMp_[rangeDesc.deviceId].totalAllocated, rangeDesc.size);
        memUsageMp_[rangeDesc.deviceId].allocSize = -rangeDesc.size;
        torchNpuRecord.memoryUsage = memUsageMp_[rangeDesc.deviceId];
        torchNpuRecord.pid = Utility::GetPid();
        torchNpuRecord.tid = Utility::GetTid();
        if (!EventReport::Instance(CommType::SOCKET).ReportTorchNpu(torchNpuRecord)) {
            CLIENT_ERROR_LOG("Report Npu Data Failed");
        }
    }
}

}