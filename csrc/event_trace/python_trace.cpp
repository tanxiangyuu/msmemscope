// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "python_trace.h"
#include <string>
#include <iostream>
#include <vector>
#include "kernel_hooks/runtime_hooks.h"
#include "event_report.h"
#include "trace_manager/event_trace_manager.h"

namespace MemScope {

bool PythonTrace::IsIgnore(std::string funcName)
{
    for (auto s : ignorePyFunc_) {
        if (s == funcName) {
            return true;
        }
    }
    return false;
}

void PythonTrace::RecordPyCall(const std::string& funcHash, const std::string& funcInfo, uint64_t timestamp)
{
    uint64_t tid = Utility::GetTid();
    if (throw_[tid]) {
        return;
    }
    int32_t devId = GD_INVALID_NUM;
    if (!GetDevice(&devId) || devId == GD_INVALID_NUM) {
        LOG_ERROR("[trace] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
    }
    std::shared_ptr<TraceEvent> event = std::make_shared<TraceEvent>();
    event->startTs = timestamp ? timestamp : Utility::GetTimeNanoseconds();
    event->hash = funcHash;
    event->info = funcInfo;
    event->pid = Utility::GetPid();
    event->tid = tid;
    event->device = std::to_string(devId);
    std::string funcName = funcHash.substr(funcHash.find(":") + 1);
    if (IsIgnore(funcName) && !throw_[tid]) {
        throw_[tid] = true;
    }
    frameStack_[tid].push(event);
}

void PythonTrace::DumpTraceEvent(std::shared_ptr<TraceEvent>& event)
{
    if (event->device == "N/A") {
        sharedEventLists_.push_back(event);
        return ;
    }
    auto it = handlerMap_.find(event->device);
    if (it == handlerMap_.end()) {
        handlerMap_.insert({event->device, MakeDataHandler(GetConfig(), DataType::PYTHON_TRACE_EVENT, event->device)});
    }
    handlerMap_[event->device]->Write(event);
}

void PythonTrace::RecordCCall(std::string funcHash, std::string funcInfo)
{
    uint64_t tid = Utility::GetTid();
    if (throw_[tid]) {
        return;
    }
    int32_t devId = GD_INVALID_NUM;
    if (!GetDevice(&devId) || devId == GD_INVALID_NUM) {
        LOG_ERROR("[trace] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
    }
    std::shared_ptr<TraceEvent> event = std::make_shared<TraceEvent>();
    event->startTs = Utility::GetTimeNanoseconds();
    event->hash = funcHash;
    event->info = funcInfo;
    event->pid = Utility::GetPid();
    event->tid = tid;
    event->device = std::to_string(devId);
    frameStack_[tid].push(event);
}

void PythonTrace::RecordReturn(std::string funcHash, std::string funcInfo)
{
    uint64_t tid = Utility::GetTid();
    if (!frameStack_[tid].empty()) {
        auto event = frameStack_[tid].top();
        if (funcHash == event->hash) {
            throw_[tid] = false;
            event->endTs = Utility::GetTimeNanoseconds();
            DumpTraceEvent(event);
            frameStack_[tid].pop();
        } else if (throw_[tid] == false) {
            int32_t devId = GD_INVALID_NUM;
            if (!GetDevice(&devId) || devId == GD_INVALID_NUM) {
                LOG_ERROR("[trace] RT_ERROR_INVALID_VALUE, " + std::to_string(devId));
            }
            std::shared_ptr<TraceEvent> event = std::make_shared<TraceEvent>(
                0, Utility::GetTimeNanoseconds(), tid, Utility::GetPid(), std::to_string(devId), funcInfo, funcHash);
            DumpTraceEvent(event);
        }
    }
}

void callback(const std::string& hash, const std::string& info, PyTraceType what, uint64_t timestamp)
{
    if (!EventTraceManager::Instance().IsTracingEnabled()) {
        return;
    }
    switch (what) {
        case PyTraceType::PYCALL: {
            PythonTrace::GetInstance().RecordPyCall(hash, info, timestamp);
            break;
        }
        case PyTraceType::PYRETURN: {
            PythonTrace::GetInstance().RecordReturn(hash, info);
            break;
        }
        case PyTraceType::CCALL: {
            PythonTrace::GetInstance().RecordCCall(hash, info);
            break;
        }
        case PyTraceType::CRETURN: {
            PythonTrace::GetInstance().RecordReturn(hash, info);
            break;
        }
        default:
            break;
    }
}

void PythonTrace::Start()
{
    if (Utility::GetPyVersion() < Utility::Version("3.9")) {
        std::cout << "[msmemscope] Warn: The current Python version is below 3.9, python trace cannot be enabled."
                  << std::endl;
        return;
    }
    bool expected{false};
    bool active = active_.compare_exchange_strong(expected, true);
    if (!active) {
        std::cout << "[msmemscope] Warn: There is already an active PythonTracer. Refusing to register profile functions."
                  << std::endl;
        return;
    }
    if (!Utility::IsPyInterpRepeInited()) {
        return;
    }
    Utility::PyInterpGuard stat;
    Utility::RegisterTraceCb(callback);
}

void PythonTrace::Stop()
{
    if (!active_) {
        std::cout << "[msmemscope] Warn: The tracer is not start." << std::endl;
        return;
    }
    if (!Utility::IsPyInterpRepeInited()) {
        return;
    }
    Utility::PyInterpGuard stat;
    Utility::UnRegisterTraceCb();
    for (auto &p : frameStack_) {
        while (!p.second.empty()) {
            DumpTraceEvent(p.second.top());
            p.second.pop();
        }
    }
    active_ = false;
}

bool PythonTrace::IsTraceActive()
{
    return active_;
}

}