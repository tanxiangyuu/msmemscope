// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 
#ifndef TRACE_EVENT_H
#define TRACE_EVENT_H

#include <cstdint>
#include <string>

#include "data.h"

namespace Leaks {

class TraceEvent : public DataBase {
public:
    TraceEvent() : DataBase(DataType::PYTHON_TRACE_EVENT) {}
    TraceEvent(const TraceEvent&) = default;
    TraceEvent& operator=(const TraceEvent&) = default;
    TraceEvent(
        uint64_t startTs,
        uint64_t endTs,
        uint64_t tid,
        uint64_t pid,
        const std::string& info,
        const std::string& hash
    ) : DataBase(DataType::PYTHON_TRACE_EVENT), startTs(startTs), endTs(endTs),
        tid(tid), pid(pid), info(info), hash(hash) {}

    uint64_t startTs = 0;
    uint64_t endTs = 0;
    uint64_t tid;
    uint64_t pid;
    std::string info;
    std::string hash;
};

}

#endif