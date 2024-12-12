// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef ANALYZER_H
#define ANALYZER_H

#include <iostream>
#include "dump_record.h"
#include "framework/config_info.h"

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
    ~Analyzer();
    void Do(const ClientId &clientId, const EventRecord &record);
    void LeakAnalyze();
private:
    AnalysisConfig config_;
    DumpRecord dump_;
    std::unordered_map<ClientId, MemoryRecordTable> memtables_{};
    void CreateMemTables(const ClientId &clientId);
    void Record(const ClientId &clientId, const EventRecord &record);
    void RecordMalloc(const ClientId &clientId, const MemOpRecord &memrecord);
    void RecordFree(const ClientId &clientId, const MemOpRecord &memrecord);
    void CheckLeak(const size_t clientId);
};
}
#endif