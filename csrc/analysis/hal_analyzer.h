// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef HAL_ANALYZER_H
#define HAL_ANALYZER_H

#include <unordered_map>
#include "module_info.h"
#include "record_info.h"
#include "config_info.h"
#include "host_injection/core/Communication.h"

namespace Leaks {
/*
 * HalAnalyzer类主要功能：
 * 1. 维护halmemalloc/halmemfree操作记录表
   2. 分析hal侧内存使用问题，泄漏问题
*/

enum class AddrStatus : uint8_t {
    FREE_ALREADY = 0U,
    FREE_WAIT,
};

struct HalMemInfo {
    int32_t deviceId;
    AddrStatus addrStatus;
};

using MemoryRecordTable = std::unordered_map<uint64_t, HalMemInfo>;

class HalAnalyzer {
public:
    static HalAnalyzer& GetInstance(Config config);
    bool Record(const ClientId &clientId, const EventRecord &record);
    bool Record(const ClientId &clientId, const RecordBase &record);
private:
    explicit HalAnalyzer(Config config);
    ~HalAnalyzer();
    HalAnalyzer(const HalAnalyzer&) = delete;
    HalAnalyzer& operator=(const HalAnalyzer&) = delete;
    HalAnalyzer(HalAnalyzer&& other) = delete;
    HalAnalyzer& operator=(HalAnalyzer&& other) = delete;

    std::unordered_map<ClientId, MemoryRecordTable> memtables_{};
    bool IsHalAnalysisEnable();
    bool CreateMemTables(const ClientId &clientId);
    void RecordMalloc(const ClientId &clientId, const MemOpRecord memrecord);
    void RecordFree(const ClientId &clientId, const MemOpRecord memrecord);
    void LeakAnalyze();
    void CheckLeak(const size_t clientId);
    Config config_;
};

}

#endif