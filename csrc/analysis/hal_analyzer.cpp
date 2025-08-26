// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "hal_analyzer.h"
#include "utility/log.h"
#include "bit_field.h"

namespace Leaks {

HalAnalyzer& HalAnalyzer::GetInstance(Config config)
{
    static HalAnalyzer analyzer(config);
    return analyzer;
}

HalAnalyzer::HalAnalyzer(Config config)
{
    config_ = config;
    return;
}

bool HalAnalyzer::IsHalAnalysisEnable()
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

    // 非默认采集模式，关闭分析功能
    if (config_.collectMode == static_cast<uint8_t>(CollectMode::DEFERRED)) {
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

bool HalAnalyzer::CreateMemTables(const ClientId &clientId)
{
    if (memtables_.find(clientId) != memtables_.end()) {
        return true;
    }
    LOG_INFO("[client %u]: Start Record hal Memory.", clientId);
    MemoryRecordTable memrecordtable{};
    auto result = memtables_.emplace(clientId, memrecordtable);
    if (result.second) {
        return true;
    }
    return false;
}

void HalAnalyzer::RecordMalloc(const ClientId &clientId, const MemOpRecord memrecord)
{
    uint64_t memkey = memrecord.addr;
    // malloc操作需解析当前moduleId
    bool foundModule = false;
    std::string modulename = "INVLID_MOUDLE_ID";
    if (MODULE_HASH_TABLE.find(memrecord.modid) != MODULE_HASH_TABLE.end()) {
        modulename = MODULE_HASH_TABLE.find(memrecord.modid)->second;
        foundModule = true;
    }
    if (!foundModule) {
        LOG_WARN("[client %u][device: %ld]: Malloc operator did not find %d Module in index %u malloc record.",
            clientId, memrecord.devId, memrecord.modid, memrecord.recordIndex);
    }

    if (memtables_[clientId].find(memkey) != memtables_[clientId].end()) {
        if ((memtables_[clientId].find(memkey)->second.addrStatus == AddrStatus::FREE_WAIT)) {
            LOG_WARN(
                "[client %u]: server already has malloc record in addr: 0x%lx ,", clientId, memrecord.addr);
            LOG_WARN("[client %u]: but now malloc again in index: %u, addr: 0x%lx, size: %u, space: %u",
                clientId, memrecord.recordIndex, memrecord.addr, memrecord.memSize, memrecord.space);
        }
    } else {
        HalMemInfo halMemInfo{};
        memtables_[clientId].emplace(memkey, halMemInfo);
    }
    memtables_[clientId][memkey].deviceId = memrecord.devId;
    memtables_[clientId][memkey].addrStatus = AddrStatus::FREE_WAIT;
}

void HalAnalyzer::RecordFree(const ClientId &clientId, const MemOpRecord memrecord)
{
    uint64_t memkey = memrecord.addr;
    auto it = memtables_[clientId].find(memkey);
    if (it != memtables_[clientId].end()) {
        if (it->second.addrStatus == AddrStatus::FREE_WAIT) {
            memtables_[clientId][memkey].addrStatus = AddrStatus::FREE_ALREADY;
        } else {
            LOG_WARN("[client %u]: Double free operator found for malloc operation : addr: 0x%lx",
                clientId, memrecord.addr);
        }
    } else {
            LOG_WARN("[client %u]: No matching malloc operation found for free operator: addr: 0x%lx",
                clientId, memrecord.addr);
    }
}

bool HalAnalyzer::Record(const ClientId &clientId, const RecordBase &record)
{
    // 判断是否满足功能开启条件
    if (!IsHalAnalysisEnable()) {
        return true;
    }
    auto memRecord = static_cast<const MemOpRecord&>(record);
    // 目前不处理CPU侧数据
    if (memRecord.devType == DeviceType::CPU) {
        return true;
    }
    if (!CreateMemTables(clientId)) {
        LOG_ERROR("[client %u]: Create hal Memory table failed.", clientId);
        return false;
    }
    if (memRecord.subtype == RecordSubType::MALLOC) {
        RecordMalloc(clientId, memRecord);
        return true;
    } else if (memRecord.subtype == RecordSubType::FREE) {
        RecordFree(clientId, memRecord);
        return true;
    }
    return false;
}

void HalAnalyzer::CheckLeak(const size_t clientId)
{
    bool foundLeaks = false;
    if (memtables_.find(clientId) != memtables_.end()) {
        for (const auto& pair :memtables_[clientId]) {
            if (pair.second.addrStatus != AddrStatus::FREE_ALREADY) {
                foundLeaks = true;
                LOG_WARN("[client %u]: Leak memory in Malloc operator, addr: 0x%lx", clientId, pair.first);
            }
        }
    }
    if (!foundLeaks) {
        LOG_INFO("[client %u]: There is no hal leak memory.", clientId);
    }
}

void HalAnalyzer::LeakAnalyze()
{
    if (!IsHalAnalysisEnable()) {
        return;
    }

    if (memtables_.empty()) {
        LOG_ERROR("No memory records available.");
    } else {
        for (const auto& pair :memtables_) {
            CheckLeak(pair.first);
        }
    }

    return;
}

HalAnalyzer::~HalAnalyzer()
{
    try {
        LeakAnalyze();
    } catch (const std::exception &ex) {
        std::cerr << "HalAnalyzer destructor catch exception: " << ex.what();
    }
}

}