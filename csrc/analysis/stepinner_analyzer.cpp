// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "stepinner_analyzer.h"
#include <cstring>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include "mstx_analyzer.h"
#include "utility/log.h"
#include "utility/utils.h"
#include "config_info.h"
#include "bit_field.h"

namespace Leaks {

StepInnerAnalyzer &StepInnerAnalyzer::GetInstance(Config config)
{
    static StepInnerAnalyzer analyzer(config);
    return analyzer;
}

StepInnerAnalyzer::StepInnerAnalyzer(Config config)
{
    config_ = config;

    auto func = std::bind(&StepInnerAnalyzer::ReceiveMstxMsg, this, std::placeholders::_1);
    MstxAnalyzer::Instance().Subscribe(MstxEventSubscriber::STEP_INNER_ANALYZER, func);

    return;
}

bool StepInnerAnalyzer::CreateTables(const DeviceId &deviceId)
{
    if (npuMemUsages_.find(deviceId) != npuMemUsages_.end()) {
        return true;
    }
    NpuMemUsage npumemusage{};
    LOG_INFO("[device %ld]: Start Record npu Memory.", deviceId);
    auto result = npuMemUsages_.emplace(deviceId, npumemusage);
    if (result.second) {
        return true;
    }
    return false;
}

bool StepInnerAnalyzer::CreateMstxTables(const DeviceId &deviceId)
{
    if (mstxTables_.find(deviceId) != mstxTables_.end()) {
        return true;
    }
    LOG_INFO("[device %ld]: Start Record mstx-npu info.", deviceId);
    MstxRecordTable mstxrecordtable{};
    auto result = mstxTables_.emplace(deviceId, mstxrecordtable);
    if (result.second) {
        return true;
    }
    return false;
}

bool StepInnerAnalyzer::CreateLeakSumTables(const DeviceId &deviceId)
{
    if (leakMemSums_.find(deviceId) != leakMemSums_.end()) {
        return true;
    }
    LOG_INFO("[device %ld]: Start Record leak memory info.", deviceId);
    LeakSumsTable leaksumstable{};
    auto result = leakMemSums_.emplace(deviceId, leaksumstable);
    if (result.second) {
        return true;
    }
    return false;
}

bool StepInnerAnalyzer::IsStepInnerAnalysisEnable()
{
    // 确认analysis设置中是否包含泄漏分析
    BitField<decltype(config_.analysisType)> analysisType(config_.analysisType);
    if (!(analysisType.checkBit(static_cast<size_t>(AnalysisType::LEAKS_ANALYSIS)))) {
        return false;
    }
    // 当开启--steps时，关闭所有分析功能
    if (config_.stepList.stepCount!=0) {
        return false;
    }

    // 当malloc和free采集并非都开启时，关闭分析功能
    BitField<decltype(config_.eventType)> eventType(config_.eventType);
    if (!(eventType.checkBit(static_cast<size_t>(EventType::ALLOC_EVENT))) ||
        !(eventType.checkBit(static_cast<size_t>(EventType::FREE_EVENT)))) {
        return false;
    }
    return true;
}
    
std::size_t LeakMemKeyHash::operator()(const LeakMemKey &leakKey) const
{
    return std::hash<uint64_t>()(leakKey.torchNpuPtr) ^ std::hash<uint64_t>()(leakKey.leakStepId);
}

bool LeakMemKey::operator==(const LeakMemKey& other) const
{
    return torchNpuPtr == other.torchNpuPtr && leakStepId == other.leakStepId;
}

void StepInnerAnalyzer::SetStepId(const DeviceId &deviceId, const uint64_t &stepId)
{
    npuMemUsages_[deviceId].mstxStep = stepId;
}

bool StepInnerAnalyzer::SkipCheck(const NpuMemInfo &npuMemInfo)
{
    // stepId为0和1，即step 1及之前申请的内存，风险低暂不告警
    if (npuMemInfo.stepId <= skipSteps_) {
        return true;
    }
    return false;
}

void StepInnerAnalyzer::CheckNpuLeak(const DeviceId &deviceId, const uint64_t stepId)
{
    for (const auto& pair :npuMemUsages_[deviceId].mempooltable) {
        if (SkipCheck(pair.second)) {
            continue;
        }
        if (pair.second.duration < durationThreshold_) {
            continue;
        }

        std::string memoryPoolType = "";
        switch (pair.second.type) {
            case RecordType::ATB_MEMORY_POOL_RECORD:
                memoryPoolType = "ATB memory pool";
                break;
            case RecordType::TORCH_NPU_RECORD:
                memoryPoolType = "Pytorch memory pool";
                break;
            case RecordType::MINDSPORE_NPU_RECORD:
                memoryPoolType = "Mindspore memory pool";
                break;
            default:
                memoryPoolType = "Unknown memory pool";
                LOG_ERROR("Undefined memorypool type!");
                break;
        }

        LOG_WARN(
            "[npu %d][step %lu]: ptr: %llx has last for %lu steps. Please check if there is leaks in %s.",
            deviceId, stepId, pair.first, pair.second.duration, memoryPoolType.c_str());
        if (!CreateLeakSumTables(deviceId)) {
            LOG_ERROR("[device %ld]: Create leaksums table failed.", deviceId);
            return;
        }
        if (leakMemSums_[deviceId].find(LeakMemKey(pair.first, pair.second.stepId)) ==
            leakMemSums_[deviceId].end()) {
            LeakInfo leakInfo{};
            leakMemSums_[deviceId].emplace(LeakMemKey(pair.first, pair.second.stepId), leakInfo);
        }

        leakMemSums_[deviceId][LeakMemKey(pair.first, pair.second.stepId)].kernelIndex =
            pair.second.kernelIndex;
        leakMemSums_[deviceId][LeakMemKey(pair.first, pair.second.stepId)].leakSize =
            npuMemUsages_[deviceId].mempooltable[pair.first].memSize;
        leakMemSums_[deviceId][LeakMemKey(pair.first, pair.second.stepId)].memoryPoolType = memoryPoolType;
    }
    return;
}

void StepInnerAnalyzer::NotifyTraceRecord(const int32_t &devId, const MemPoolRecord &memPoolRecord)
{
    uint64_t ptr = memPoolRecord.memoryUsage.ptr;
    if (npuMemUsages_[devId].mempooltable[ptr].duration >= (durationThreshold_ + 1)
        && npuMemUsages_[devId].mempooltable[ptr].stepId > skipSteps_
    ) {
        TorchMemLeakInfo info{
            devId,
            npuMemUsages_[devId].mempooltable[ptr].kernelIndex,
            memPoolRecord.kernelIndex - npuMemUsages_[devId].mempooltable[ptr].kernelIndex,
            ptr,
            -memPoolRecord.memoryUsage.allocSize
        };
        TraceRecord::GetInstance().ProcessTorchMemLeakInfo(info);
    }
    return;
}

void StepInnerAnalyzer::UpdateAllocated(const DeviceId &deviceId, const int64_t &totalAllocated)
{
    if (!IsStepInnerAnalysisEnable()) {
        return;
    }
    // 当step为0和1时，allocated尚未稳定不进行更新
    if (npuMemUsages_[deviceId].mstxStep <= skipSteps_) {
        return;
    }
    // 初值为0，Step开始，第一次更新
    if (npuMemUsages_[deviceId].stepMaxAllocated == 0) {
        npuMemUsages_[deviceId].stepMaxAllocated = totalAllocated;
        npuMemUsages_[deviceId].stepMinAllocated = totalAllocated;
        return;
    }
    if (totalAllocated > npuMemUsages_[deviceId].stepMaxAllocated) {
        npuMemUsages_[deviceId].stepMaxAllocated = totalAllocated;
    }
    if (totalAllocated < npuMemUsages_[deviceId].stepMinAllocated) {
        npuMemUsages_[deviceId].stepMinAllocated = totalAllocated;
    }
    return;
}

void StepInnerAnalyzer::CheckGap(const DeviceId &deviceId)
{
    // 当step为0和1时，allocated尚未稳定不进行更新
    if (npuMemUsages_[deviceId].mstxStep <= skipSteps_) {
        return;
    }
    if (npuMemUsages_[deviceId].stepMaxAllocated == 0) {
        LOG_WARN("[npu %d]: StepMaxAllocated is 0, please check!", deviceId);
        return;
    }
    double gap =
    npuMemUsages_[deviceId].stepMinAllocated / static_cast<double>(npuMemUsages_[deviceId].stepMaxAllocated);
    // 第一次计算
    if (npuMemUsages_[deviceId].maxGapInfo.minMaxAllocRatio == 0) {
        npuMemUsages_[deviceId].maxGapInfo.gapStepId = npuMemUsages_[deviceId].mstxStep;
        npuMemUsages_[deviceId].maxGapInfo.minMaxAllocRatio = gap;
        npuMemUsages_[deviceId].maxGapInfo.minAllocMemory = npuMemUsages_[deviceId].stepMinAllocated;

        npuMemUsages_[deviceId].minGapInfo.gapStepId = npuMemUsages_[deviceId].mstxStep;
        npuMemUsages_[deviceId].minGapInfo.minMaxAllocRatio = gap;
        npuMemUsages_[deviceId].minGapInfo.minAllocMemory = npuMemUsages_[deviceId].stepMinAllocated;

        // Step结束，还原初始化
        npuMemUsages_[deviceId].stepMaxAllocated = 0;
        npuMemUsages_[deviceId].stepMinAllocated = 0;
        return;
    }
    // 后续计算查看是否比值变化
    if (gap > npuMemUsages_[deviceId].maxGapInfo.minMaxAllocRatio) {
        LOG_WARN(
            "[npu %d]: Min/Max Allocated memory largest gap increases to %f, last is %f",
            deviceId, gap, npuMemUsages_[deviceId].maxGapInfo.minMaxAllocRatio);
        npuMemUsages_[deviceId].maxGapInfo.gapStepId = npuMemUsages_[deviceId].mstxStep;
        npuMemUsages_[deviceId].maxGapInfo.minMaxAllocRatio = gap;
        npuMemUsages_[deviceId].maxGapInfo.minAllocMemory = npuMemUsages_[deviceId].stepMinAllocated;
    }
    if (gap < npuMemUsages_[deviceId].minGapInfo.minMaxAllocRatio) {
        LOG_WARN(
            "[npu %d]: Min/Max Allocated memory smallest gap decreases to %f, last is %f",
            deviceId, gap, npuMemUsages_[deviceId].minGapInfo.minMaxAllocRatio);
        npuMemUsages_[deviceId].minGapInfo.gapStepId = npuMemUsages_[deviceId].mstxStep;
        npuMemUsages_[deviceId].minGapInfo.minMaxAllocRatio = gap;
        npuMemUsages_[deviceId].minGapInfo.minAllocMemory = npuMemUsages_[deviceId].stepMinAllocated;
    }
    // Step结束，还原初始化
    npuMemUsages_[deviceId].stepMaxAllocated = 0;
    npuMemUsages_[deviceId].stepMinAllocated = 0;
    return;
}

void StepInnerAnalyzer::RecordNpuMalloc(const ClientId &clientId, const DeviceId &deviceId,
    const MemPoolRecord &memPoolRecord)
{
    MemoryUsage memoryusage = memPoolRecord.memoryUsage;
    int64_t npumemptr = memoryusage.ptr;
    if ((npuMemUsages_[deviceId].mempooltable.find(npumemptr) != npuMemUsages_[deviceId].mempooltable.end())) {
        LOG_WARN(
            "[npu%d malloc][client %u]:!!! ------double malloc------!!!, ptr: %lld", deviceId, clientId, npumemptr);
    }
    NpuMemInfo npuMemInfo = {
    memPoolRecord.type, memoryusage.allocSize, memPoolRecord.timeStamp, 0, npuMemUsages_[deviceId].mstxStep,
    memPoolRecord.kernelIndex};
    npuMemUsages_[deviceId].mempooltable.emplace(npumemptr, npuMemInfo);
    UpdateAllocated(deviceId, memoryusage.totalAllocated);
    npuMemUsages_[deviceId].totalAllocated = memoryusage.totalAllocated;
    npuMemUsages_[deviceId].totalReserved = memoryusage.totalReserved;
    npuMemUsages_[deviceId].totalActive = memoryusage.totalActive;
    return;
}

void  StepInnerAnalyzer::RecordNpuFree(const ClientId &clientId, const DeviceId &deviceId,
    const MemPoolRecord &memPoolRecord)
{
    MemoryUsage memoryusage = memPoolRecord.memoryUsage;
    int64_t npumemptr = memoryusage.ptr;
    if ((npuMemUsages_[deviceId].mempooltable.find(npumemptr) == npuMemUsages_[deviceId].mempooltable.end())) {
        LOG_WARN(
            "[npu%d free][client %u]:!!! ------free error------!!!, ptr: %lld", deviceId, clientId, npumemptr);
    }

    // 在释放时获取跨多个Step释放内存信息
    NotifyTraceRecord(deviceId, memPoolRecord);

    npuMemUsages_[deviceId].mempooltable.erase(npumemptr);
    UpdateAllocated(deviceId, memoryusage.totalAllocated);
    npuMemUsages_[deviceId].totalAllocated = memoryusage.totalAllocated;
    npuMemUsages_[deviceId].totalReserved = memoryusage.totalReserved;
    npuMemUsages_[deviceId].totalActive = memoryusage.totalActive;
    return;
}

void StepInnerAnalyzer::AddDuration(const DeviceId &deviceId)
{
    if (npuMemUsages_.find(deviceId) == npuMemUsages_.end()) {
        LOG_ERROR("[device %ld]: No npu memorypool record!", deviceId);
        return;
    }
    for (auto& pair :npuMemUsages_[deviceId].mempooltable) {
        pair.second.duration += 1;
    }
    return;
}

int64_t StepInnerAnalyzer::GetNowAllocated(const DeviceId &deviceId)
{
    if (npuMemUsages_.find(deviceId) == npuMemUsages_.end()) {
        LOG_ERROR("[device %ld]: No npu memorypool record!", deviceId);
        return 0;
    }
    return npuMemUsages_[deviceId].totalAllocated;
}

bool StepInnerAnalyzer::Record(const ClientId &clientId, const EventRecord &record)
{
    // 当开启--steps时，关闭所有step内分析功能
    if (!IsStepInnerAnalysisEnable()) {
        return true;
    }
    MemPoolRecord memPoolRecord = record.record.memPoolRecord;
    DeviceId deviceId = memPoolRecord.memoryUsage.deviceIndex;
    if (!CreateTables(deviceId)) {
        LOG_ERROR("[device %ld]: Create npu Memory table failed.", deviceId);
        return false;
    }
    // 目前不处理BLOCK_FREE操作
    if (memPoolRecord.memoryUsage.dataType == static_cast<uint8_t>(MemActionType::MALLOC)) {
        RecordNpuMalloc(clientId, deviceId, memPoolRecord);
    } else if (memPoolRecord.memoryUsage.dataType == static_cast<uint8_t>(MemActionType::FREE)) {
        RecordNpuFree(clientId, deviceId, memPoolRecord);
    }
    return true;
}

void StepInnerAnalyzer::ReceiveMstxMsg(const MstxRecord &mstxRecord)
{
    auto deviceId = mstxRecord.devId;
    auto stepId = mstxRecord.stepId;
    if (!IsStepInnerAnalysisEnable()) {
        return;
    }
    MarkType markType = mstxRecord.markType;
    if (!CreateMstxTables(deviceId) || !CreateTables(deviceId)) {
        LOG_WARN("[device %ld]: Create mstx-npu table failed.", deviceId);
        return;
    }
    if (markType == MarkType::RANGE_START_A) {
        if (strcmp(mstxRecord.markMessage, "step start") != 0) {
            return;
        }
        int64_t startAllocated = GetNowAllocated(deviceId);
        LOG_INFO("[npu %ld][step %llu][start]: ------Start totalAllocated: %lld------",
            deviceId, stepId, startAllocated);
        if (mstxTables_[deviceId].find(stepId) == mstxTables_[deviceId].end()) {
            StepInfo stepInfo{};
            mstxTables_[deviceId].emplace(stepId, stepInfo);
        }
        mstxTables_[deviceId][stepId].totalAllocated = startAllocated;
        mstxTables_[deviceId][stepId].rangeId = mstxRecord.rangeId;
        SetStepId(deviceId, stepId);
        AddDuration(deviceId);
    } else if (markType == MarkType::RANGE_END) {
        // 如果是end看stepid和rangeid是否在table中
        if (mstxTables_[deviceId].find(stepId) == mstxTables_[deviceId].end() ||
            mstxTables_[deviceId][stepId].rangeId != mstxRecord.rangeId) {
            return;
        }
        int64_t endAllocated = GetNowAllocated(deviceId);
        LOG_INFO("[npu %ld][step %llu][end]: ------End totalAllocated: %lld------",
            deviceId, stepId, endAllocated);
        int64_t startAllocated = mstxTables_[deviceId][stepId].totalAllocated;
        // step1不考虑前后内存不一致
        if (stepId == 1) {
            return;
        }
        if (startAllocated == endAllocated) {
            LOG_INFO("[npu %ld][step %llu][end]: ------No leaks------", deviceId, stepId);
        } else {
            LOG_INFO("[npu %ld][step %llu][end]: ------leaks------", deviceId, stepId);
        }
        CheckNpuLeak(deviceId, stepId);
        CheckGap(deviceId);
    }
    return;
}

 void StepInnerAnalyzer::ReportLeak(const DeviceId &deviceId)
 {
    std::cout<< "====== ERROR: Detected memory leaks on device " <<
        deviceId << " ======" << std::endl;

    // 依照step排序
    std::vector<std::pair<LeakMemKey, LeakInfo>> leakVec(
        leakMemSums_[deviceId].begin(), leakMemSums_[deviceId].end());
    auto leakCompare = [](const std::pair<LeakMemKey, LeakInfo> &a, const std::pair<LeakMemKey, LeakInfo> &b) {
            return a.first.leakStepId <= b.first.leakStepId;
    };
    std::sort(leakVec.begin(), leakVec.end(), leakCompare);
    // 输出泄漏信息总结
    uint64_t leakInfoCounts = 0;
    long double leakSizeSums = 0;
    for (const auto& pair :leakVec) {
        printf("Direct %s leak of %f Mb(s) at 0x%lx in kernel_%lu at step %lu.\n",
            pair.second.memoryPoolType.c_str(),
            (pair.second.leakSize / static_cast<double>(BYTE_TO_MB)),
            pair.first.torchNpuPtr,
            pair.second.kernelIndex,
            pair.first.leakStepId);
        leakInfoCounts++;
        long double leakTempSize = static_cast<long double>(pair.second.leakSize);
        leakSizeSums = Utility::GetAddResult(leakTempSize, leakSizeSums);
    }
    std::cout << "====== SUMMARY: " << leakSizeSums / BYTE_TO_MB << " Mb(s) leaked in " <<
        leakInfoCounts << " allocation(s) ======" << std::endl;
    return;
}

void StepInnerAnalyzer::ReportGap(const DeviceId &deviceId)
{
    // 打屏为保持格式统一需调整小数精度，打屏结束后还原
    int currentPrecision = std::cout.precision();
    int outputPrecision = 4;
    int outputWidth = 25;
    std::cout << "======= Memory Gap Analysis of Device " << deviceId << " =======" << std::endl;
    std::cout << "\t"
              << std::setw(outputWidth) << std::left << "MinAlloc/MaxAlloc(%)"
              << std::setw(outputWidth) << std::left << "MinAllocMem(MB)"
              << std::setw(outputWidth) << std::left << "StepId"
              << std::endl;
    std::cout << "MinGap\t"
              << std::fixed << std::setprecision(outputPrecision)
              << std::setw(outputWidth) << std::left
              << npuMemUsages_[deviceId].minGapInfo.minMaxAllocRatio * PERCENT_SCALE_FACTOR
              << std::setw(outputWidth) << std::left
              << npuMemUsages_[deviceId].minGapInfo.minAllocMemory / static_cast<double>(BYTE_TO_MB)
              << std::setw(outputWidth) << std::left
              << npuMemUsages_[deviceId].minGapInfo.gapStepId
              << std::endl;
    std::cout << "MaxGap\t"
              << std::setw(outputWidth) << std::left
              << npuMemUsages_[deviceId].maxGapInfo.minMaxAllocRatio * PERCENT_SCALE_FACTOR
              << std::setw(outputWidth) << std::left
              << npuMemUsages_[deviceId].maxGapInfo.minAllocMemory / static_cast<double>(BYTE_TO_MB)
              << std::setw(outputWidth) << std::left
              << npuMemUsages_[deviceId].maxGapInfo.gapStepId
              << std::setprecision(currentPrecision)
              << std::endl;
}

StepInnerAnalyzer::~StepInnerAnalyzer()
{
    MstxAnalyzer::Instance().UnSubscribe(MstxEventSubscriber::STEP_INNER_ANALYZER);

    if (!IsStepInnerAnalysisEnable()) {
        return;
    }
    if (!leakMemSums_.empty()) {
        for (const auto& pair :leakMemSums_) {
            ReportLeak(pair.first);
        }
    }
    // 输出内存波动与模型的权重内存大小
    for (auto &device : npuMemUsages_) {
        ReportGap(device.first);
    }

    return;
}

}