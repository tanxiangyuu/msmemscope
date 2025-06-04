// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "python_trace.h"
#include <string>
#include <iostream>
#include <vector>
#include "kernel_hooks/runtime_hooks.h"
#include "event_report.h"

namespace Leaks {

bool PythonTrace::IsIgnore(std::string funcName)
{
    for (auto s : ignorePyFunc_) {
        if (s == funcName) {
            return true;
        }
    }
    return false;
}

void PythonTrace::RecordPyCall(std::string funcHash, std::string funcInfo, uint64_t timeStamp)
{
    uint64_t tid = Utility::GetTid();
    if (throw_[tid]) {
        return;
    }
    TraceEvent event{};
    event.startTs = timeStamp ? timeStamp : Utility::GetTimeNanoseconds();
    event.hash = funcHash;
    event.info = funcInfo;
    event.pid = Utility::GetPid();
    event.tid = tid;
    std::string funcName = funcHash.substr(funcHash.find(":") + 1);
    if (IsIgnore(funcName) && !throw_[tid]) {
        throw_[tid] = true;
    }
    frameStack_[tid].push(event);
}

bool PythonTrace::DumpTraceEvent(TraceEvent &event)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!handler_->Init()) {
        return false;
    }
    CallStackString emptyStack {};
    return handler_->Write(&event, emptyStack);
}

void PythonTrace::RecordCCall(std::string funcHash, std::string funcInfo)
{
    uint64_t tid = Utility::GetTid();
    if (throw_[tid]) {
        return;
    }
    TraceEvent event{};
    event.startTs = Utility::GetTimeNanoseconds();
    event.hash = funcHash;
    event.info = funcInfo;
    event.pid = Utility::GetPid();
    event.tid = tid;
    frameStack_[tid].push(event);
}

void PythonTrace::RecordReturn(std::string funcHash, std::string funcInfo)
{
    uint64_t tid = Utility::GetTid();
    if (!frameStack_[tid].empty()) {
        auto event = frameStack_[tid].top();
        if (funcHash == event.hash) {
            throw_[tid] = false;
            event.endTs = Utility::GetTimeNanoseconds();
            DumpTraceEvent(event);
            frameStack_[tid].pop();
        } else if (throw_[tid] == false) {
            TraceEvent event{0, Utility::GetTimeNanoseconds(), tid, Utility::GetPid(), funcInfo, funcHash};
            DumpTraceEvent(event);
        }
    }
}

void callback(std::string hash, std::string info, PyTraceType what, uint64_t timeStamp)
{
    switch (what) {
        case PyTraceType::PYCALL: {
            PythonTrace::GetInstance().RecordPyCall(hash, info, timeStamp);
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
        std::cout << "[msleaks] Warn: The current Python version is below 3.9, python trace cannot be enabled."
                  << std::endl;
        return;
    }
    bool expected{false};
    bool active = active_.compare_exchange_strong(expected, true);
    if (!active) {
        std::cout << "[msleaks] Warn: There is already an active PythonTracer. Refusing to register profile functions."
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
        std::cout << "[msleaks] Warn: The tracer is not start." << std::endl;
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

}