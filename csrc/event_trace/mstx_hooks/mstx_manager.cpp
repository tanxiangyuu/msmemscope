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
}

void MstxManager::ReportRegionsUnregister(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const *desc)
{
    auto tracer = MemoryPoolTraceManager::GetInstance().GetMemoryPoolTracer(domain);
    if (tracer) {
        return tracer->Release(domain, desc);
    }
}

}