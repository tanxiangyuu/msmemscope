// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "stepinner_analyzer.h"
#include <cstring>
#include <iostream>
#include <algorithm>
#include "mstx_analyzer.h"

namespace Leaks {

StepInnerAnalyzer::StepInnerAnalyzer(const AnalysisConfig &config)
{
    config_ = config;
    // 当开启--steps时，关闭所有分析功能，只保留记录torch_npu信息的功能
    if (config_.stepList.stepCount!=0) {
        IsStepInnerAnalysisEnable_ = false;
    }
}

bool StepInnerAnalyzer::CreateTables(const DeviceId &deviceId)
{
    if (npumemusages_.find(deviceId) != npumemusages_.end()) {
        return true;
    }
    NpuMemUsage npumemusage{};
    Utility::LogInfo("[device %ld]: Start Record npu Memory.", deviceId);
    auto result = npumemusages_.emplace(deviceId, npumemusage);
    if (result.second) {
        return true;
    }
    return false;
}

bool StepInnerAnalyzer::CreateMstxTables(const DeviceId &deviceId)
{
    if (mstxtables_.find(deviceId) != mstxtables_.end()) {
        return true;
    }
    Utility::LogInfo("[device %ld]: Start Record mstx-npu info.", deviceId);
    MstxRecordTable mstxrecordtable{};
    auto result = mstxtables_.emplace(deviceId, mstxrecordtable);
    if (result.second) {
        return true;
    }
    return false;
}

bool StepInnerAnalyzer::CreateLeakSumTables(const DeviceId &deviceId)
{
    if (leakmemusums_.find(deviceId) != leakmemusums_.end()) {
        return true;
    }
    Utility::LogInfo("[device %ld]: Start Record leak memory info.", deviceId);
    LeakMemSums leakmemusums{};
    auto result = leakmemusums_.emplace(deviceId, leakmemusums);
    if (result.second) {
        return true;
    }
    return false;
}
    
std::size_t LeakMemKeyHash::operator()(const LeakMemKey &l_key) const {
    return std::hash<uint64_t>()(l_key.torchNpuPtr) ^ std::hash<uint64_t>()(l_key.leakRangeId);
}

bool LeakMemKey::operator==(const LeakMemKey& other) const
{
    return torchNpuPtr == other.torchNpuPtr && leakRangeId == other.leakRangeId;
}

void StepInnerAnalyzer::SetRangeId(const DeviceId &deviceId, const uint64_t &rangeId)
{
    npumemusages_[deviceId].mstxRange = rangeId;
}

bool StepInnerAnalyzer::SkipCheck(const NpuMemInfo &npuMemInfo)
{
    // rangeId为0和1，即step 1之前申请的内存，风险低暂不告警
    if (npuMemInfo.rangeId == 0 || npuMemInfo.rangeId == 1) {
        return true;
    }
    return false;
}

void StepInnerAnalyzer::CheckNpuLeak(const DeviceId &deviceId, const uint64_t rangeId)
{
    for (const auto& pair :npumemusages_[deviceId].mempooltable) {
        if (SkipCheck(pair.second)) {
            continue;
        }
        if (pair.second.duration >= durationThreshold_) {
            Utility::LogWarn(
                "[npu %d][step %lu]: ptr: %ld has last for %lu steps. Please check if there is memory leaks.",
                deviceId, rangeId, pair.first, pair.second.duration);
            if (!CreateLeakSumTables(deviceId)) {
                Utility::LogError("[device %ld]: Create leaksums table failed.", deviceId);
                return;
            }
            leakmemusums_[deviceId].leaksumstable[LeakMemKey(pair.first, pair.second.rangeId)].duration =
                pair.second.duration;
            leakmemusums_[deviceId].leaksumstable[LeakMemKey(pair.first, pair.second.rangeId)].leakSize =
                npumemusages_[deviceId].mempooltable[pair.first].memSize;
        }
    }
    return;
}

void StepInnerAnalyzer::NotifyTraceRecord(const int32_t &devId, const TorchNpuRecord &torchNpuRecord)
{
    uint64_t ptr = torchNpuRecord.memoryUsage.ptr;
    if (npumemusages_[devId].mempooltable[ptr].duration >= (durationThreshold_ + 1)
        && npumemusages_[devId].mempooltable[ptr].rangeId > skipSteps_
    ) {
        TorchMemLeakInfo info{
            devId,
            npumemusages_[devId].mempooltable[ptr].timestamp,
            torchNpuRecord.timeStamp - npumemusages_[devId].mempooltable[ptr].timestamp,
            ptr,
            -torchNpuRecord.memoryUsage.allocSize
        };
        TraceRecord::GetInstance().ProcessTorchMemLeakInfo(info);
    }
    return;
}

void StepInnerAnalyzer::RecordNpuMalloc(const ClientId &clientId, const DeviceId &deviceId,
    const TorchNpuRecord &torchnpuRecord)
{
    MemoryUsage memoryusage = torchnpuRecord.memoryUsage;
    uint64_t npumemptr = memoryusage.ptr;
    if (IsStepInnerAnalysisEnable_ &&
        (npumemusages_[deviceId].mempooltable.find(npumemptr) != npumemusages_[deviceId].mempooltable.end())) {
        Utility::LogError(
            "[npu%d malloc][client %u]:!!! ------double malloc------!!!, ptr: %lld", deviceId, clientId, npumemptr);
    }
    Utility::LogInfo(
        "[npu%d malloc][client %u]: index:%llu, totalAllocated: %lld, allocSize: %lld, ptr: %lld, step: %llu",
        deviceId,
        clientId,
        torchnpuRecord.recordIndex,
        memoryusage.totalAllocated,
        memoryusage.allocSize,
        npumemptr,
        npumemusages_[deviceId].mstxRange);
    npumemusages_[deviceId].mempooltable[npumemptr].memSize = memoryusage.allocSize;
    npumemusages_[deviceId].mempooltable[npumemptr].timestamp = torchnpuRecord.timeStamp;
    npumemusages_[deviceId].mempooltable[npumemptr].duration = 0;
    npumemusages_[deviceId].mempooltable[npumemptr].rangeId = npumemusages_[deviceId].mstxRange;
    npumemusages_[deviceId].totalAllocated = memoryusage.totalAllocated;
    npumemusages_[deviceId].totalReserved = memoryusage.totalReserved;
    npumemusages_[deviceId].totalActive = memoryusage.totalActive;
    return;
}

void  StepInnerAnalyzer::RecordNpuFree(const ClientId &clientId, const DeviceId &deviceId,
    const TorchNpuRecord &torchnpuRecord)
{
    MemoryUsage memoryusage = torchnpuRecord.memoryUsage;
    uint64_t npumemptr = memoryusage.ptr;
    if (IsStepInnerAnalysisEnable_ &&
         (npumemusages_[deviceId].mempooltable.find(npumemptr) == npumemusages_[deviceId].mempooltable.end())) {
        Utility::LogError(
            "[npu%d free][client %u]:!!! ------free error------!!!, ptr: %lld", deviceId, clientId, npumemptr);
    }
    Utility::LogInfo(
        "[npu%d free][client %u]: index:%llu, totalAllocated: %lld, allocSize: %lld, ptr: %lld, duration: %llu steps",
        deviceId,
        clientId,
        torchnpuRecord.recordIndex,
        memoryusage.totalAllocated,
        memoryusage.allocSize,
        npumemptr,
        npumemusages_[deviceId].mempooltable[npumemptr].duration);
    
    // 在释放时获取跨多个Step释放内存信息
    NotifyTraceRecord(deviceId, torchnpuRecord);

    npumemusages_[deviceId].mempooltable.erase(npumemptr);
    npumemusages_[deviceId].totalAllocated = memoryusage.totalAllocated;
    npumemusages_[deviceId].totalReserved = memoryusage.totalReserved;
    npumemusages_[deviceId].totalActive = memoryusage.totalActive;
    return;
}

void StepInnerAnalyzer::AddDuration(const DeviceId &deviceId)
{
    if (npumemusages_.find(deviceId) == npumemusages_.end()) {
        Utility::LogError("[device %ld]: No npu memorypool record!", deviceId);
        return;
    }
    for (auto& pair :npumemusages_[deviceId].mempooltable) {
        pair.second.duration += 1;
    }
    return;
}

int64_t StepInnerAnalyzer::GetNowAllocated(const DeviceId &deviceId)
{
    if (npumemusages_.find(deviceId) == npumemusages_.end()) {
        Utility::LogError("[device %ld]: No npu memorypool record!", deviceId);
        return 0;
    }
    return npumemusages_[deviceId].totalAllocated;
}

bool StepInnerAnalyzer::Record(const ClientId &clientId, const EventRecord &record)
{
    TorchNpuRecord torchnpuRecord = record.record.torchNpuRecord;
    DeviceId deviceId = torchnpuRecord.memoryUsage.deviceIndex;
    if (!CreateTables(deviceId)) {
        Utility::LogError("[device %ld]: Create npu Memory table failed.", deviceId);
        return false;
    }
    // 目前不处理FREE操作
    if (torchnpuRecord.memoryUsage.dataType == static_cast<uint8_t>(MemActionType::MALLOC)) {
        RecordNpuMalloc(clientId, deviceId, torchnpuRecord);
    } else if (torchnpuRecord.memoryUsage.dataType == static_cast<uint8_t>(MemActionType::BLOCK_FREE)) {
        RecordNpuFree(clientId, deviceId, torchnpuRecord);
    }
    return true;
}

void StepInnerAnalyzer::ReceiveMstxMsg(const DeviceId &deviceId, const uint64_t &rangeId, const MstxRecord &mstxRecord)
{
    if (!IsStepInnerAnalysisEnable_) {
        return;
    }
    MarkType markType = mstxRecord.markType;
    if (!CreateMstxTables(deviceId)) {
        Utility::LogError("[device %ld]: Create mstx-npu table failed.", deviceId);
        return;
    }
    if (!CreateTables(deviceId)) {
        Utility::LogError("[device %ld]: Create npu Memory table failed.", deviceId);
        return;
    }
    if (markType == MarkType::RANGE_START_A) {
        // 看是否有固化的语句来判断是否要分析
        if (strcmp(mstxRecord.markMessage, "[stepInnerAnalyzer] step start") != 0) {
            return;
        }
        int64_t startAllocated = GetNowAllocated(deviceId);
        Utility::LogInfo("[npu %ld][step %llu][start]: ------Start totalAllocated: %lld------",
            deviceId, rangeId, startAllocated);
        mstxtables_[deviceId][rangeId] = startAllocated;
        SetRangeId(deviceId, rangeId);
        AddDuration(deviceId);
    } else if (markType == MarkType::RANGE_END) {
        // 如果是end看rangeid是否在table中
        if (mstxtables_[deviceId].find(rangeId) == mstxtables_[deviceId].end()) {
            return;
        }
        int64_t endAllocated = GetNowAllocated(deviceId);
        Utility::LogInfo("[npu %ld][step %llu][end]: ------End totalAllocated: %lld------",
            deviceId, rangeId, endAllocated);
        
        int64_t startAllocated = mstxtables_[deviceId][rangeId];
        // step1不考虑前后内存不一致
        if (rangeId == 1) {
            return;
        }
        if (startAllocated == endAllocated) {
            Utility::LogInfo("[npu %ld][step %llu][end]: ------No leaks------", deviceId, rangeId);
        } else {
            Utility::LogError("[npu %ld][step %llu][end]: ------leaks------", deviceId, rangeId);
            std::cout<< "====== Warning: LeakCheck: detected torch_npu memory growth with steps on device " <<
                deviceId << " ======" << std::endl;
            std::cout << "[npu " << deviceId << "][step " << rangeId << "]: Start totalAllocated=" <<
            startAllocated << " byte(s), End totalAllocated=" << endAllocated << " byte(s)." << std::endl;
        }
        CheckNpuLeak(deviceId, rangeId);
    }
    return;
}

 void StepInnerAnalyzer::ReportLeak(const DeviceId &deviceId)
 {
    std::cout<< "====== Warning: LeakCheck: detected torch_npu memory leaks on device " <<
        deviceId << " ======" << std::endl;

    // 依照step排序
    std::vector<std::pair<LeakMemKey, LeakInfo>> leakVec(
        leakmemusums_[deviceId].leaksumstable.begin(), leakmemusums_[deviceId].leaksumstable.end());
    auto leakCompare = [](const std::pair<LeakMemKey, LeakInfo> &a, const std::pair<LeakMemKey, LeakInfo> &b) {
        if (a.first.leakRangeId != b.first.leakRangeId) {
            return a.first.leakRangeId < b.first.leakRangeId;
        } else {
            return a.second.duration < b.second.duration;
        }
    };
    std::sort(leakVec.begin(), leakVec.end(), leakCompare);
    // 输出泄漏信息总结
    uint64_t leakInfoCounts = 0;
    long double leakSizeSums = 0;
    for (const auto& pair :leakVec) {
        printf("[npu %d]: ptr: %ld has last from step %lu to step %lu. Leak size: %f Mb(s).\n",
            deviceId,
            pair.first.torchNpuPtr,
            pair.first.leakRangeId,
            pair.first.leakRangeId + pair.second.duration,
            (pair.second.leakSize) / 1048576.0
            );
        leakInfoCounts++;
        leakSizeSums += pair.second.leakSize;
    }
    std::cout << "===== SUMMARY: Total " << leakInfoCounts << " leaks, " <<
        leakSizeSums / 1048576.0 << " Mb(s) on device " << deviceId << " =====" << std::endl;
    return;
}


StepInnerAnalyzer::~StepInnerAnalyzer()
{
    if (!IsStepInnerAnalysisEnable_) {
        return;
    }
    if (leakmemusums_.empty()) {
        std::cout << "[msleaks] There is no torch_npu leaks." << std::endl;
    } else {
        for (const auto& pair :leakmemusums_) {
            ReportLeak(pair.first);
        }
    }

    return;
}

}