// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef ANALYZER_H
#define ANALYZER_H

#include "framework/config_info.h"
#include "framework/record_info.h"

namespace Leaks {

// 用于内存哈希比较的Key
class  MemOpRecordKey{
public:
    //MemOpSpace space_; // 内存操作空间：device还是host
    uint64_t addr_; // 地址
    explicit MemOpRecordKey(const uint64_t &addr);
    bool operator==(const MemOpRecordKey &other) const;

};

// 计算MemOpRecordKey哈希值
struct MemOpRecordKeyHash{
    std::size_t operator()(const MemOpRecordKey &memrecordkey) const;
};

// 内存哈希表类
class MemoryHashTable {
public:
    void Record(const EventRecord &record);
    void CheckLeak();
private:
    std::unordered_map<MemOpRecordKey, MemOpType, MemOpRecordKeyHash> table;    
};

// Analyzer类主要用于将单条解析信息分发给合适的分析工具
class Analyzer {
public:
    explicit Analyzer(const AnalysisConfig &config);
    void Do(const EventRecord &record);
    void LeakAnalyze();
private:
    AnalysisConfig config_;
    MemoryHashTable memhashtable;
};

}

#endif
