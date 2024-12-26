// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include <cstring>
#include "mstx_analyzer.h"
#include "stepinner_analyzer.h"

namespace Leaks {

StepInnerAnalyzer::StepInnerAnalyzer(const AnalysisConfig &config)
{
    config_ = config;
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

void StepInnerAnalyzer::SetRangeId(const DeviceId &deviceId, const uint64_t &rangeId)
{
    npumemusages_[deviceId].mstxRange = rangeId;
}

bool StepInnerAnalyzer::SkipCheck(const LeakInfo &leakInfo)
{
    // rangeId为0，即step 1之前申请的内存，风险低暂不告警
    if (leakInfo.rangeId == 0) {
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
                "[npu %d][rangeid %llu]: ptr: %lld has last for %llu steps. Please check if there is memory leaks.",
                deviceId, rangeId, pair.first, pair.second.duration);
        }
    }
    return;
}

void StepInnerAnalyzer::RecordNpuMalloc(const ClientId &clientId, const DeviceId &deviceId,
    const TorchNpuRecord &torchnpuRecord)
{
    MemoryUsage memoryusage = torchnpuRecord.memoryUsage;
    uint64_t npumemptr = memoryusage.ptr;

    if (npumemusages_[deviceId].mempooltable.find(npumemptr) != npumemusages_[deviceId].mempooltable.end()) {
        Utility::LogError("!!! ------double malloc------!!!, ptr: %lld", npumemptr);
    }
    Utility::LogInfo(
        "[npu%d malloc][client %u]: index:%llu, totalAllocated: %lld, allocSize: %lld, ptr: %lld, rangid: %llu",
        deviceId,
        clientId,
        torchnpuRecord.recordIndex,
        memoryusage.totalAllocated,
        memoryusage.allocSize,
        npumemptr,
        npumemusages_[deviceId].mstxRange);
    npumemusages_[deviceId].mempooltable[npumemptr].recordIndex = torchnpuRecord.recordIndex;
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

    if (npumemusages_[deviceId].mempooltable.find(npumemptr) == npumemusages_[deviceId].mempooltable.end()) {
        Utility::LogError("!!! ------free error------!!!, ptr: %lld", npumemptr);
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

void StepInnerAnalyzer::Record(const ClientId &clientId, const EventRecord &record)
{
    TorchNpuRecord torchnpuRecord = record.record.torchNpuRecord;
    DeviceId deviceId = torchnpuRecord.memoryUsage.deviceIndex;
    if (!CreateTables(deviceId)) {
        Utility::LogError("[device %ld]: Create npu Memory table failed.", deviceId);
        return;
    }
    // 目前不处理Block_free操作
    if (torchnpuRecord.memoryUsage.dataType == 0) {
        RecordNpuMalloc(clientId, deviceId, torchnpuRecord);
    } else if (torchnpuRecord.memoryUsage.dataType == 1) {
        RecordNpuFree(clientId, deviceId, torchnpuRecord);
    }
    return;
}

void StepInnerAnalyzer::ReceiveMstxMsg(const DeviceId &deviceId, const uint64_t &rangeId, const MstxRecord &mstxRecord)
{
    MarkType markType = mstxRecord.markType;
    if (!CreateMstxTables(deviceId)) {
        Utility::LogError("[device %ld]: Create mstx-npu table failed.", deviceId);
        return;
    }
    if (markType == MarkType::RANGE_START_A) {
        // 看是否有固化的语句来判断是否要分析
        if (strcmp(mstxRecord.markMessage, "[stepInnerAnalyzer] step start") != 0) {
            return;
        }
        int64_t startAllocated = GetNowAllocated(deviceId);
        Utility::LogInfo("[npu %ld][rangeid %llu][start]: ------Start totalAllocated: %lld------",
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
        Utility::LogInfo("[npu %ld][rangeid %llu][end]: ------End totalAllocated: %lld------",
            deviceId, rangeId, endAllocated);
        
        int64_t startAllocated = mstxtables_[deviceId][rangeId];
        // step1不考虑前后内存不一致
        if (rangeId == 1) {
            return;
        }
        if (startAllocated == endAllocated) {
            Utility::LogInfo("[npu %ld][rangeid %llu][end]: ------No leaks------", deviceId, rangeId);
        } else {
            Utility::LogError("[npu %ld][rangeid %llu][end]: ------leaks------", deviceId, rangeId);
        }
        CheckNpuLeak(deviceId, rangeId);
    }
    return;
}

}