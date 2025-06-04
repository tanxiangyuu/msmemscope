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

namespace Leaks {

struct TraceEvent {
    TraceEvent() = default;
    TraceEvent(const TraceEvent&) = default;
    TraceEvent& operator=(const TraceEvent&) = default;

    uint64_t startTs;
    uint64_t endTs;
    uint64_t tid;
    uint64_t pid;
    std::string info;
    std::string hash;
};

class PythonTrace {
public:
    static PythonTrace& GetInstance()
    {
        static PythonTrace instance;
        return instance;
    }
    PythonTrace(const PythonTrace&) = delete;
    PythonTrace& operator=(const PythonTrace&) = delete;
    void RecordPyCall(std::string funcHash, std::string funcInfo, uint64_t timeStamp);
    void RecordCCall(std::string funcHash, std::string funcInfo);
    void RecordReturn(std::string funcHash, std::string funcInfo);
    void Start();
    void Stop();
private:
    bool DumpTraceEvent(TraceEvent &event);
    bool IsIgnore(std::string func);
    PythonTrace()
    {
        auto config = EventReport::Instance(CommType::SOCKET).GetConfig();
        prefix_ = "python_trace_" + std::to_string(Utility::GetPid()) + "_";
        dirPath_ = std::string(config.outputDir) + "/dump";
    }
    ~PythonTrace()
    {
        if (dataFile_ != nullptr) {
            fclose(dataFile_);
            dataFile_ = nullptr;
        }
    }
    std::unordered_map<uint64_t, std::stack<TraceEvent>> frameStack_;
    std::atomic<bool> active_{false};
    std::unordered_map<uint64_t, bool> throw_;
    std::mutex mutex_;
    FILE *dataFile_ = nullptr;
    std::string prefix_;
    std::string dirPath_;
    std::vector<std::string> ignorePyFunc_ = {"__torch_dispatch__"};
};
void callback(std::string hash, std::string info, PyTraceType what, uint64_t timeStamp);
}

#endif