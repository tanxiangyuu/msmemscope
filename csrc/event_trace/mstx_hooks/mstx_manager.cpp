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
#include "memory_pool_trace/pta_memory_pool_trace.h"
#include "describe_trace.h"

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
    auto config = EventReport::Instance(CommType::SOCKET).GetConfig();
    std::string pyStack;
    if (config.enablePyStack) {
        Utility::GetPythonCallstack(config.pyStackDepth, pyStack);
    }
    TLVBlockType pyStackType = pyStack.empty() ? TLVBlockType::SKIP : TLVBlockType::CALL_STACK_PYTHON;
    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<MstxRecord>(
        TLVBlockType::MARK_MESSAGE, msg, pyStackType, pyStack);
 
    MstxRecord* record = buffer.Cast<MstxRecord>();
    record->markType = MarkType::MARK_A;
    record->rangeId = onlyMarkId_;
    record->streamId = streamId;

    if (!EventReport::Instance(CommType::SOCKET).ReportMark(buffer)) {
        CLIENT_ERROR_LOG("Report Mark FAILED");
    }
}

// 组装Range开始打点信息
uint64_t MstxManager::ReportRangeStart(const char* msg, int32_t streamId)
{
    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<MstxRecord>(TLVBlockType::MARK_MESSAGE, msg);
 
    MstxRecord* record = buffer.Cast<MstxRecord>();
    record->markType = MarkType::RANGE_START_A;
    record->streamId = streamId;
    record->rangeId = GetRangeId();
    const TLVBlock* msgTlv = GetTlvBlock(*record, TLVBlockType::MARK_MESSAGE);
    std::string mstxMsgString = msgTlv == nullptr ? "N/A" : msgTlv->data;
    if (!EventReport::Instance(CommType::SOCKET).ReportMark(buffer)) {
        CLIENT_ERROR_LOG("Report Mark FAILED");
    }
    return record->rangeId;
}

// 组装Range结束打点信息
void MstxManager::ReportRangeEnd(uint64_t id)
{
    std::string msg = "Range end from id " + std::to_string(id);
    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<MstxRecord>(TLVBlockType::MARK_MESSAGE, msg);
    MstxRecord* record = buffer.Cast<MstxRecord>();
    record->markType = MarkType::RANGE_END;
    record->streamId = -1;
    record->rangeId = id;

    if (!EventReport::Instance(CommType::SOCKET).ReportMark(buffer)) {
        CLIENT_ERROR_LOG("Report Mark FAILED");
    }
}

uint64_t MstxManager::GetRangeId()
{
    return rangeId_++;
}
// MSTX针对内存池的分析功能 这里进行代码重构 和上面的打点功能剥离
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
    if (std::string(domainName) == "msleaks") {
        // PTA会进行多次内存池注册 已经注册过就返回之前注册的domain
        MemoryPoolTraceManager::GetInstance().RegisterMemoryPoolTracer("msleaks", &PTAMemoryPoolTrace::GetInstance());
        return MemoryPoolTraceManager::GetInstance().CreateDomain(domainName);
    }
    return nullptr;
}

mstxMemHeapHandle_t MstxManager::ReportHeapRegister(mstxDomainHandle_t domain, mstxMemHeapDesc_t const *desc)
{
    auto tracer = MemoryPoolTraceManager::GetInstance().GetMemoryPoolTracer(domain);
    if (tracer) {
        return tracer->Allocate(domain, desc);
    }
    return nullptr;
}

void MstxManager::ReportHeapUnregister(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap)
{
    auto tracer = MemoryPoolTraceManager::GetInstance().GetMemoryPoolTracer(domain);
    if (tracer) {
        return tracer->Deallocate(domain, heap);
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
        int devId = rangeDescArray[i].deviceId;
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
            TLVBlockType::ADDR_OWNER, owner, cStackType, stack.cStack, pyStackType, stack.pyStack);
        MemPoolRecord* record = buffer.Cast<MemPoolRecord>();
        record->type = RecordType::TORCH_NPU_RECORD;
        record->memoryUsage = memUsageMp_[devId];
        if (!EventReport::Instance(CommType::SOCKET).ReportMemPoolRecord(buffer)) {
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
        mstxMemVirtualRangeDesc_t rangeDesc = regionHandleMp_[desc->refArray[i].pointer];
        memUsageMp_[rangeDesc.deviceId].dataType = 1;
        memUsageMp_[rangeDesc.deviceId].deviceIndex = rangeDesc.deviceId;
        memUsageMp_[rangeDesc.deviceId].ptr = reinterpret_cast<int64_t>(rangeDesc.ptr);
        memUsageMp_[rangeDesc.deviceId].totalAllocated =
            Utility::GetSubResult(memUsageMp_[rangeDesc.deviceId].totalAllocated, rangeDesc.size);
        memUsageMp_[rangeDesc.deviceId].allocSize = rangeDesc.size;
        std::string owner = "";
        TLVBlockType cStackType = stack.cStack.empty() ? TLVBlockType::SKIP : TLVBlockType::CALL_STACK_C;
        TLVBlockType pyStackType = stack.pyStack.empty() ? TLVBlockType::SKIP : TLVBlockType::CALL_STACK_PYTHON;
        RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<MemPoolRecord>(
            TLVBlockType::ADDR_OWNER, owner, cStackType, stack.cStack, pyStackType, stack.pyStack);
        MemPoolRecord* record = buffer.Cast<MemPoolRecord>();
        record->type = RecordType::TORCH_NPU_RECORD;
        record->memoryUsage = memUsageMp_[rangeDesc.deviceId];
        if (!EventReport::Instance(CommType::SOCKET).ReportMemPoolRecord(buffer)) {
            CLIENT_ERROR_LOG("Report Npu Data Failed");
        }
    }
}

}