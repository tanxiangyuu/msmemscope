// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef STEPINNER_ANALYZER_H
#define STEPINNER_ANALYZER_H

#include "trace_record.h"
#include "config_info.h"
#include "comm_def.h"

namespace Leaks {
/*
 * StepInnerAnalyzer类主要功能：
 * 1. 维护npu内存池的申请释放表，记录未释放的内存持续step时间
   2. 维护观察mstx的mstx-npu表，用于分析step内的内存泄漏
*/

using DeviceId = int32_t;
using StepId = uint64_t;
constexpr uint64_t BYTE_TO_MB = 1024 * 1024;
constexpr double  PERCENT_SCALE_FACTOR = 100.0;

const std::unordered_map<RecordType, std::string> RecordTypeToString = {
    {RecordType::ATB_MEMORY_POOL_RECORD, "ATB memory pool"},
    {RecordType::PTA_CACHING_POOL_RECORD, "Pytorch Caching memory pool"},
    {RecordType::MINDSPORE_NPU_RECORD, "Mindspore memory pool"}
};

struct StepInfo {
    std::unordered_map<RecordType, int64_t> stepAllocTable;
    uint64_t rangeId = 0;  // rangeId唯一标识，用于判断是否为固化信息的打点信息
};

using MstxRecordTable = std::unordered_map<StepId, StepInfo>;

enum class MemActionType : uint8_t {
    MALLOC = 0,
    FREE = 1,
    BLOCK_FREE = 2
};

struct  LeakMemKey  {
    uint64_t ptr;
    RecordType type;
    uint64_t leakStepId;
    LeakMemKey(uint64_t p, RecordType t, uint64_t id) : ptr(p), type(t), leakStepId(id) {}
    bool operator==(const LeakMemKey& other) const;
};

struct LeakInfo {
    int64_t leakSize;
    uint64_t kernelIndex;
};

struct LeakMemKeyHash {
    std::size_t operator()(const LeakMemKey& leakKey) const;
};

using LeakSumsTable = std::unordered_map<LeakMemKey, LeakInfo, LeakMemKeyHash>;

// 单条内存操作信息
struct NpuMemInfo {
    RecordType type;
    int64_t memSize;
    uint64_t timestamp;
    uint64_t duration;      // 目前经历的duration
    uint64_t stepId;        // 来自哪个mstx的stepId
    uint64_t kernelIndex;   // 处于哪个event中
};

struct GapInfo  {
    uint64_t gapStepId = 0;             // 记录计算比值时的stepId
    double  minMaxAllocRatio = 0;       // 最大allocated内存和最小allocated内存的比值
    int64_t minAllocMemory = 0;         // 最小allocated内存的值
};

// 用于记录不同类型内存池内的占用状态
struct MemoryPoolStatus {
    int64_t totalAllocated = 0;
    int64_t totalReserved = 0;
    int64_t totalActive = 0;
    int64_t stepMaxAllocated = 0;
    int64_t stepMinAllocated = 0;
    GapInfo maxGapInfo;    // 记录动态内存和静态内存比值最大的信息
    GapInfo minGapInfo;    // 记录动态内存和静态内存比值最小的信息
};

struct NpuMemKey {
    uint64_t ptr;
    RecordType memType;
    NpuMemKey(uint64_t p, RecordType t) : ptr(p), memType(t) {}
    bool operator==(const NpuMemKey& other) const;
};

struct NpuMemKeyHash {
    std::size_t operator()(const NpuMemKey& memKey) const;
};

struct NpuMemUsage {
    std::unordered_map<NpuMemKey, NpuMemInfo, NpuMemKeyHash> poolOpTable; // 维护内存池具体操作的表
    std::unordered_map<RecordType, MemoryPoolStatus> poolStatusTable;      // 维护内存池占用状态的表
    uint64_t mstxStep = 0; // 用于更新当前到哪一个step，并将其应用于表中的stepId属性。
};

class StepInnerAnalyzer {
public:
    static StepInnerAnalyzer &GetInstance(Config config);
    bool Record(const ClientId &clientId, const RecordBase &record);
private:
    explicit StepInnerAnalyzer(Config config);
    ~StepInnerAnalyzer();
    StepInnerAnalyzer(const StepInnerAnalyzer&) = delete;
    StepInnerAnalyzer& operator=(const StepInnerAnalyzer&) = delete;
    StepInnerAnalyzer(StepInnerAnalyzer&& other) = delete;
    StepInnerAnalyzer& operator=(StepInnerAnalyzer&& other) = delete;
    
    const std::string& GetMemoryPoolName(const RecordType &poolType);
    void UpdateMstxTable(const MstxRecord &mstxRecord, const RecordType &poolType, const int64_t &startAllocated);
    void ReceiveMstxMsg(const MstxRecord &mstxRecord);
    void UpdateAllocated(const DeviceId &deviceId, const RecordType &poolType, const int64_t &totalAllocated);
    void AddDuration(const DeviceId &deviceId);
    void SetStepId(const DeviceId &deviceId, const uint64_t &stepId);
    void CheckGap(const DeviceId &deviceId);
    void CheckNpuLeak(const DeviceId &deviceId, const uint64_t stepId);
    void NotifyTraceRecord(const int32_t &devId, const MemPoolRecord &memPoolRecord);
    bool CreateMstxTables(const DeviceId &deviceId);
    bool CreateTables(const DeviceId &deviceId);
    bool CreateLeakSumTables(const DeviceId &deviceId);
    void RecordNpuMalloc(const ClientId &clientId, const DeviceId &deviceId, const MemPoolRecord &memPoolRecord);
    void RecordNpuFree(const ClientId &clientId, const DeviceId &deviceId, const MemPoolRecord &memPoolRecord);
    bool SkipCheck(const NpuMemInfo &npuMemInfo);
    void ReportLeak(const DeviceId &deviceId);
    void ReportGap(const DeviceId &deviceId);
    bool IsStepInnerAnalysisEnable();
    std::unordered_map<DeviceId, NpuMemUsage> npuMemUsages_{};
    std::unordered_map<DeviceId, MstxRecordTable> mstxTables_{};
    std::unordered_map<DeviceId, LeakSumsTable> leakMemSums_{};
    uint64_t durationThreshold_ = 1;  // 设置警告阈值, 可由用户更改
    uint64_t skipSteps_ = 1;
    Config config_;
};

}

#endif