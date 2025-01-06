// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef STEPINNER_ANALYZER_H
#define STEPINNER_ANALYZER_H

#include "host_injection/core/LocalProcess.h"
#include "analyzer_base.h"
#include "trace_record.h"

namespace Leaks {
/*
 * StepInnerAnalyzer类主要功能：
 * 1. 维护npu内存池的申请释放表，记录未释放的内存持续step时间
   2. 维护观察mstx的mstx-npu表，用于分析step内的内存泄漏
*/

using DeviceId = int32_t;
using RangeId = uint64_t;
using TotalAllocated = int64_t;

using MstxRecordTable = std::unordered_map<RangeId, TotalAllocated>;

enum class MemActionType : uint8_t {
    MALLOC = 0,
    FREE = 1,
    BLOCK_FREE = 2
};

struct LeakInfo {
    uint64_t timestamp;
    uint64_t duration; // 目前经历的duration
    uint64_t rangeId; // 来自哪个mstx的rangeId
};

struct NpuMemUsage {
    std::unordered_map<uint64_t, LeakInfo> mempooltable;
    int64_t totalAllocated = 0;
    int64_t totalReserved = 0;
    int64_t totalActive = 0;
    uint64_t mstxRange = 0; // 用于更新当前到哪一个step，并将其应用于表中的rangeId属性。
};

class StepInnerAnalyzer : public AnalyzerBase {
public:
    explicit StepInnerAnalyzer(const AnalysisConfig &config);
    bool Record(const ClientId &clientId, const EventRecord &record) override;
    void ReceiveMstxMsg(const DeviceId &deviceId, const uint64_t &rangeId, const MstxRecord &mstxRecord) override;
    void AddDuration(const DeviceId &deviceId);
    void SetRangeId(const DeviceId &deviceId, const uint64_t &rangeId);
    int64_t GetNowAllocated(const DeviceId &deviceId);
    void CheckNpuLeak(const DeviceId &deviceId, const uint64_t rangeId);
    void NotifyTraceRecord(const int32_t &devId, const TorchNpuRecord &torchnpuRecord);
private:
    std::unordered_map<DeviceId, NpuMemUsage> npumemusages_{};
    std::unordered_map<DeviceId, MstxRecordTable> mstxtables_{};
    bool CreateMstxTables(const DeviceId &deviceId);
    bool CreateTables(const DeviceId &deviceId);
    void RecordNpuMalloc(const ClientId &clientId, const DeviceId &deviceId, const TorchNpuRecord &torchnpuRecord);
    void RecordNpuFree(const ClientId &clientId, const DeviceId &deviceId, const TorchNpuRecord &torchnpuRecord);
    bool SkipCheck(const LeakInfo &leakInfo);
    int64_t durationThreshold_ = 1;  // 设置警告阈值, 可由用户更改
    uint64_t skipSteps_ = 1;
    AnalysisConfig config_;
};

}

#endif