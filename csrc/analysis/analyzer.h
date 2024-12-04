// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef ANALYZER_H
#define ANALYZER_H

#include <vector>
#include <iostream>
#include <unordered_map>
#include "framework/config_info.h"
#include "framework/record_info.h"
#include "host_injection/core/Communication.h"

namespace Leaks {

enum class AddrStatus : uint8_t {
    FREE_ALREADY = 0U,
    FREE_WAIT,
};

using MemoryRecordTable = std::unordered_map<uint64_t, AddrStatus>;

// Analyzer类主要用于将单条解析信息分发给合适的分析工具
class Analyzer {
public:
    explicit Analyzer(const AnalysisConfig &config);
    void Do(const ClientId &clientId, const EventRecord &record);
    void LeakAnalyze();
    ~Analyzer();
private:
    AnalysisConfig config_;
    std::unordered_map<ClientId, MemoryRecordTable> memtables_{};
    void CreateMemTables(const ClientId &clientId);
    void Record(const ClientId &clientId, const EventRecord &record);
    void RecordMalloc(const ClientId &clientId, const MemOpRecord memrecord);
    void RecordFree(const ClientId &clientId, const MemOpRecord memrecord);
    void CheckLeak(const size_t clientId);
};

}

#endif
