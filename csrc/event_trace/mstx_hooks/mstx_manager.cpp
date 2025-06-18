// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "mstx_manager.h"
#include <cstring>
#include "securec.h"
#include "call_stack.h"
#include "event_report.h"
#include "record_info.h"
#include "log.h"
#include "bit_field.h"
#include "aten_manager.h"
#include "memory_pool_trace/memory_pool_trace_manager.h"
#include "memory_pool_trace/atb_memory_pool_trace.h"
#include "memory_pool_trace/mindspore_memory_pool_trace.h"

namespace Leaks {

// 组装普通打点信息
void MstxManager::ReportMarkA(const char* msg, int32_t streamId)
{
    // 处理aten算子上报信息
    if (msg && strncmp(msg, ATEN_MSG, strlen(ATEN_MSG)) == 0) {
        const char* atenMsg = msg + strlen(ATEN_MSG);
        AtenManager::GetInstance().ProcessMsg(atenMsg, streamId);
        return ;
    }
    MstxRecord record;
    record.markType = MarkType::MARK_A;
    record.rangeId = onlyMarkId_;
    record.streamId = streamId;

    if (strncpy_s(record.markMessage, sizeof(record.markMessage), msg, sizeof(record.markMessage) - 1) != EOK) {
        CLIENT_ERROR_LOG("strncpy_s FAILED");
    }
    record.markMessage[sizeof(record.markMessage) - 1] = '\0';
    auto config = EventReport::Instance(CommType::SOCKET).GetConfig();
    std::string cStack;
    std::string pyStack;
    if (config.enablePyStack) {
        Utility::GetPythonCallstack(config.pyStackDepth, pyStack);
    }
    CallStackString stack{cStack, pyStack};
    if (!EventReport::Instance(CommType::SOCKET).ReportMark(record, stack)) {
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
    CallStackString stack;
    if (!EventReport::Instance(CommType::SOCKET).ReportMark(record, stack)) {
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
    CallStackString stack;
    if (!EventReport::Instance(CommType::SOCKET).ReportMark(record, stack)) {
        CLIENT_ERROR_LOG("Report Mark FAILED");
    }
}

uint64_t MstxManager::GetRangeId()
{
    return rangeId_++;
}

mstxDomainHandle_t MstxManager::ReportDomainCreateA(char const *domainName)
{
    // 后续收编所有通过MSTX打点的内存池trace
    if (std::string(domainName) == "atb") {
        if (MemoryPoolTraceManager::GetInstance().RegisterMemoryPoolTracer("atb", &ATBMemoryPoolTrace::GetInstance())) {
            return MemoryPoolTraceManager::GetInstance().CreateDomain(domainName);
        }
    }
    if (std::string(domainName) == "mindsporeMemPool") {
        if (MemoryPoolTraceManager::GetInstance().RegisterMemoryPoolTracer("mindsporeMemPool",
            &MindsporeMemoryPoolTrace::GetInstance())) {
            return MemoryPoolTraceManager::GetInstance().CreateDomain(domainName);
        }
    }
    if (domainName == nullptr || std::string(domainName) != "msleaks") {
        return nullptr;
    }
    return msleaksDomain_;
}

mstxMemHeapHandle_t MstxManager::ReportHeapRegister(mstxDomainHandle_t domain, mstxMemHeapDesc_t const *desc)
{
    auto tracer = MemoryPoolTraceManager::GetInstance().GetMemoryPoolTracer(domain);
    if (tracer) {
        return tracer->Allocate(domain, desc);
    }
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
    auto tracer = MemoryPoolTraceManager::GetInstance().GetMemoryPoolTracer(domain);
    if (tracer) {
        return tracer->Deallocate(domain, heap);
    }
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
    auto tracer = MemoryPoolTraceManager::GetInstance().GetMemoryPoolTracer(domain);
    if (tracer) {
        return tracer->Reallocate(domain, desc);
    }
    if (domain == nullptr || desc == nullptr || domain != msleaksDomain_) {
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
        MemPoolRecord memPoolRecord;
        int devId = rangeDescArray[i].deviceId;
        memUsageMp_[devId].dataType = 0;
        memUsageMp_[devId].deviceIndex = devId;
        memUsageMp_[devId].ptr = reinterpret_cast<int64_t>(rangeDescArray[i].ptr);
        memUsageMp_[devId].allocSize = rangeDescArray[i].size;
        memUsageMp_[devId].totalAllocated =
            Utility::GetAddResult(memUsageMp_[devId].totalAllocated, memUsageMp_[devId].allocSize);
        regionHandleMp_[rangeDescArray[i].ptr] = rangeDescArray[i];
        memPoolRecord.type = RecordType::TORCH_NPU_RECORD;
        memPoolRecord.memoryUsage = memUsageMp_[devId];
        memPoolRecord.pid = Utility::GetPid();
        memPoolRecord.tid = Utility::GetTid();
        if (!EventReport::Instance(CommType::SOCKET).ReportMemPoolRecord(memPoolRecord, stack)) {
            CLIENT_ERROR_LOG("Report Npu Data Failed");
        }
    }
}

void MstxManager::ReportRegionsUnregister(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const *desc)
{
    auto tracer = MemoryPoolTraceManager::GetInstance().GetMemoryPoolTracer(domain);
    if (tracer) {
        return tracer->Release(domain, desc);
    }
    if (domain == nullptr || desc == nullptr || domain != msleaksDomain_) {
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
            CLIENT_ERROR_LOG("Report Npu Data Failed");
        }
    }
}

}