// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 
#ifndef INEFFICIENT_ANALYZER_H
#define INEFFICIENT_ANALYZER_H

#include <atomic>
#include "analyzer_base.h"
 
namespace Leaks {

constexpr uint64_t THREHOLD = 3000;
constexpr uint64_t MAX_UNIT64 = std::numeric_limits<uint64_t>::max();
constexpr uint64_t MIN_EVENTS_NUM = 2; // state buffer中需要最少有两条内存访问记录
constexpr uint64_t LAST_EVENTS_NUM = 1;

class InefficientAnalyzer : public AnalyzerBase {
public:
    static InefficientAnalyzer& GetInstance();
    void EventHandle(std::shared_ptr<EventBase>& event, MemoryState* state) override;
    struct PidState {
        std::vector<std::shared_ptr<MemoryEvent>> apiTmp;
        std::atomic<uint64_t> apiId;
        uint64_t mallocApiTmpId;
        uint64_t freeApiTmpId;
        bool isOpStart;
    };
private:
    explicit InefficientAnalyzer();
    ~InefficientAnalyzer() override = default;
    InefficientAnalyzer(const InefficientAnalyzer&) = delete;
    InefficientAnalyzer& operator=(const InefficientAnalyzer&) = delete;
    InefficientAnalyzer(InefficientAnalyzer&& other) = delete;
    InefficientAnalyzer& operator=(InefficientAnalyzer&& other) = delete;

    void InefficientAnalysis(std::shared_ptr<MemoryEvent>& event, MemoryState* state, std::unordered_map<uint64_t, PidState>& pidStatesMap);
    void HandleOpLaunchEvent(std::shared_ptr<EventBase>& event, std::unordered_map<uint64_t, PidState>& pidStatesMap);
    void HandleMemoryEvent(std::shared_ptr<EventBase>& event, MemoryState* state, std::unordered_map<uint64_t, PidState>& pidStatesMap);
    void EarlyAllocation(std::shared_ptr<MemoryEvent>& event, MemoryState* state, std::unordered_map<uint64_t, PidState>& pidStatesMap);
    void LateDeallocation(std::shared_ptr<MemoryEvent>& event, MemoryState* state, std::unordered_map<uint64_t, PidState>& pidStatesMap);
    void TemporaryIdleness(std::shared_ptr<MemoryEvent>& event, MemoryState* state);
    void Init(const uint64_t pid, std::unordered_map<uint64_t, PidState>& pidStatesMap);
    void AddEventToTmps(const std::shared_ptr<MemoryEvent>& event, std::unordered_map<uint64_t, PidState>& pidStatesMap);
    void AddApiIdToState(std::shared_ptr<MemoryEvent>& event, MemoryState* state, std::unordered_map<uint64_t, PidState>& pidStatesMap);
    void ClassifyEventsTmp(const uint64_t pid, std::unordered_map<uint64_t, PidState>& pidStatesMap);
    void UpdateApiId(const uint64_t pid, std::unordered_map<uint64_t, PidState>& pidStatesMap);

    std::unordered_map<uint64_t, PidState> PTAPidStatesMap;
    std::unordered_map<uint64_t, PidState> ATBPidStatesMap;
};
}
 
#endif