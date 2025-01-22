// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "stepinner_analyzer.h"
#include <cstring>
#include <iostream>
#include "mstx_analyzer.h"

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

void StepInnerAnalyzer::SetStepId(const DeviceId &deviceId, const uint64_t &stepId)
{
    npumemusages_[deviceId].mstxStep = stepId;
}

bool StepInnerAnalyzer::SkipCheck(const LeakInfo &leakInfo)
{
    // stepId为0，即step 1之前申请的内存，风险低暂不告警
    if (leakInfo.stepId == 0) {
        return true;
    }
    return false;
}

void StepInnerAnalyzer::CheckNpuLeak(const DeviceId &deviceId, const uint64_t stepId)
{
    for (const auto& pair :npumemusages_[deviceId].mempooltable) {
        if (SkipCheck(pair.second)) {
            continue;
        }
        if (pair.second.duration >= durationThreshold_) {
            printf("[npu %d][stepid %lu]: ptr: %ld has last for %lu steps. Please check if there is memory leaks.\n",
                deviceId, stepId, pair.first, pair.second.duration);
        }
    }
    return;
}

void StepInnerAnalyzer::NotifyTraceRecord(const int32_t &devId, const TorchNpuRecord &torchNpuRecord)
{
    uint64_t ptr = torchNpuRecord.memoryUsage.ptr;
    if (npumemusages_[devId].mempooltable[ptr].duration >= durationThreshold_
        && npumemusages_[devId].mempooltable[ptr].stepId >= skipSteps_
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
    if (npumemusages_[deviceId].mempooltable.find(npumemptr) != npumemusages_[deviceId].mempooltable.end()) {
        Utility::LogError(
            "[npu%d malloc][client %u]:!!! ------double malloc------!!!, ptr: %lld", deviceId, clientId, npumemptr);
    }
    Utility::LogInfo(
        "[npu%d malloc][client %u]: index:%llu, totalAllocated: %lld, allocSize: %lld, ptr: %lld, rangid: %llu",
        deviceId,
        clientId,
        torchnpuRecord.recordIndex,
        memoryusage.totalAllocated,
        memoryusage.allocSize,
        npumemptr,
        npumemusages_[deviceId].mstxStep);
    npumemusages_[deviceId].mempooltable[npumemptr].timestamp = torchnpuRecord.timeStamp;
    npumemusages_[deviceId].mempooltable[npumemptr].duration = 0;
    npumemusages_[deviceId].mempooltable[npumemptr].stepId = npumemusages_[deviceId].mstxStep;
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

void StepInnerAnalyzer::ReceiveMstxMsg(const DeviceId &deviceId, const uint64_t &stepId, const MstxRecord &mstxRecord)
{
    MarkType markType = mstxRecord.markType;
    if (!CreateMstxTables(deviceId)) {
        Utility::LogError("[device %ld]: Create mstx-npu table failed.", deviceId);
        return;
    }
    if (markType == MarkType::RANGE_START_A) {
        // 看是否有固化的语句来判断是否要分析
        if (strcmp(mstxRecord.markMessage, "step start") != 0) {
            return;
        }
        int64_t startAllocated = GetNowAllocated(deviceId);
        Utility::LogInfo("[npu %ld][stepid %llu][start]: ------Start totalAllocated: %lld------",
            deviceId, stepId, startAllocated);
        mstxtables_[deviceId][stepId] = startAllocated;
        SetStepId(deviceId, stepId);
        AddDuration(deviceId);
    } else if (markType == MarkType::RANGE_END) {
        // 如果是end看stepid是否在table中
        if (mstxtables_[deviceId].find(stepId) == mstxtables_[deviceId].end()) {
            return;
        }
        int64_t endAllocated = GetNowAllocated(deviceId);
        Utility::LogInfo("[npu %ld][stepid %llu][end]: ------End totalAllocated: %lld------",
            deviceId, stepId, endAllocated);
        
        int64_t startAllocated = mstxtables_[deviceId][stepId];
        // step1不考虑前后内存不一致
        if (stepId == 1) {
            return;
        }
        if (startAllocated == endAllocated) {
            Utility::LogInfo("[npu %ld][stepid %llu][end]: ------No leaks------", deviceId, stepId);
        } else {
            std::cout << "[npu " << deviceId << "][stepid " << stepId << "][end]: ------leaks------" << std::endl;
        }
        CheckNpuLeak(deviceId, stepId);
    }
    return;
}

}