// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef ANALYZER_H
#define ANALYZER_H

#include <vector>
#include <iostream>
#include "framework/config_info.h"
#include "framework/record_info.h"
#include "host_injection/core/Communication.h"

namespace Leaks {

// 用于内存哈希比较的Key
class  MemOpRecordKey {
public:
    uint64_t addr_; // 地址
    explicit MemOpRecordKey(const uint64_t &addr) : addr_(addr){};
    bool operator==(const MemOpRecordKey &other) const;
};

// 计算MemOpRecordKey哈希值
struct MemOpRecordKeyHash {
    std::size_t operator()(const MemOpRecordKey &memrecordkey) const;
};

// 内存哈希表类
class MemoryHashTable {
public:
    void Record(const ClientId &clientId, const EventRecord &record);
    void RecordMalloc(const ClientId &clientId, const MemOpRecord memrecord, const EventRecord &record);
    void RecordFree(const ClientId &clientId, const MemOpRecord memrecord);
    void CheckLeak(const size_t clientId);
private:
    std::unordered_map<MemOpRecordKey, int32_t, MemOpRecordKeyHash> table;
};

// Analyzer类主要用于将单条解析信息分发给合适的分析工具
class Analyzer {
public:
    explicit Analyzer(const AnalysisConfig &config);
    void Do(const ClientId &clientId, const EventRecord &record);
    void LeakAnalyze();
    ~Analyzer();
private:
    AnalysisConfig config_;
    std::vector<MemoryHashTable> memtablelist{};
    MemoryHashTable& GetMemTable(const ClientId &clientId);
};

}

#endif
