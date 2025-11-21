// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#ifndef PYTHON_TRACE_H
#define PYTHON_TRACE_H

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <stack>
#include <iostream>
#include "cpython.h"
#include "config_info.h"
#include "record_info.h"
#include "utils.h"
#include "file.h"
#include "event_report.h"
#include "../csrc/analysis/data_handler.h"

namespace MemScope {

class PythonTrace {
public:
    static PythonTrace& GetInstance()
    {
        static PythonTrace instance;
        return instance;
    }
    PythonTrace(const PythonTrace&) = delete;
    PythonTrace& operator=(const PythonTrace&) = delete;
    void RecordPyCall(const std::string& funcHash, const std::string& funcInfo, uint64_t timestamp);
    void RecordCCall(std::string funcHash, std::string funcInfo);
    void RecordReturn(std::string funcHash, std::string funcInfo);
    void Start();
    void Stop();
    bool IsTraceActive();
private:
    void DumpTraceEvent(std::shared_ptr<TraceEvent>& event);
    bool IsIgnore(std::string funcName);
    PythonTrace() = default;
    ~PythonTrace() = default;
    std::unordered_map<uint64_t, std::stack<std::shared_ptr<TraceEvent>>> frameStack_;
    std::atomic<bool> active_{false};
    std::unordered_map<uint64_t, bool> throw_;
    std::string prefix_;
    std::string dirPath_;
    std::vector<std::string> ignorePyFunc_ = {"__torch_dispatch__"};
    std::unordered_map<std::string, std::unique_ptr<DataHandler>> handlerMap_;
    std::vector<std::shared_ptr<TraceEvent>> sharedEventLists_;
};
void callback(const std::string& hash, const std::string& info, PyTraceType what, uint64_t timestamp);
}

#endif