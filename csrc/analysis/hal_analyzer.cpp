// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "hal_analyzer.h"

namespace Leaks {

HalAnalyzer::HalAnalyzer(const AnalysisConfig &config)
{
    config_ = config;
}

void HalAnalyzer::ReceiveMstxMsg(const DeviceId &deviceId, const uint64_t &rangeid, const MstxRecord &mstxrecord)
{
}

bool HalAnalyzer::CreateMemTables(const ClientId &clientId)
{
    if (memtables_.find(clientId) != memtables_.end()) {
        return true;
    }
    Utility::LogInfo("[client %u]: Start Record hal Memory.", clientId);
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
    if (g_ModuleHashTable.find(memrecord.modid) != g_ModuleHashTable.end()) {
        modulename = g_ModuleHashTable.find(memrecord.modid)->second;
        foundModule = true;
    }
    if (!foundModule) {
        Utility::LogError("[client %u][device: %ld]: Malloc operator did not find %d Module in index %u malloc record.",
            clientId, memrecord.devid, memrecord.modid, memrecord.recordIndex);
    }

    Utility::LogInfo(
        "[client %u][device: %ld]: server malloc record, index: %u, addr: 0x%lx, size: %u, space: %u, module: %s",
        clientId, memrecord.devid, memrecord.recordIndex,
        memrecord.addr, memrecord.memSize, memrecord.space, modulename.c_str());

    if (memtables_[clientId].find(memkey) != memtables_[clientId].end() &&
        (memtables_[clientId].find(memkey)->second == AddrStatus::FREE_WAIT)) {
        Utility::LogError("[client %u]: server already has malloc record in addr: 0x%lx ,", clientId, memrecord.addr);
        Utility::LogError("[client %u]: but now malloc again in index: %u, addr: 0x%lx, size: %u, space: %u",
            clientId, memrecord.recordIndex, memrecord.addr, memrecord.memSize, memrecord.space);
    }
    memtables_[clientId][memkey] = AddrStatus::FREE_WAIT;
}

void HalAnalyzer::RecordFree(const ClientId &clientId, const MemOpRecord memrecord)
{
    uint64_t memkey = memrecord.addr;
    Utility::LogInfo("[client %u][device: %ld]: server free record, index: %u, addr: 0x%lx",
        clientId, memrecord.devid, memrecord.recordIndex, memrecord.addr);

    auto it = memtables_[clientId].find(memkey);
    if (it != memtables_[clientId].end()) {
        if (it->second == AddrStatus::FREE_WAIT) {
            memtables_[clientId][memkey] = AddrStatus::FREE_ALREADY;
        } else {
            Utility::LogError("[client %u]: Double free operator found for malloc operation : addr: 0x%lx",
                clientId, memrecord.addr);
        }
    } else {
            Utility::LogError("[client %u]: No matching malloc operation found for free operator: addr: 0x%lx",
                clientId, memrecord.addr);
    }
}

void HalAnalyzer::Record(const ClientId &clientId, const EventRecord &record)
{
    if (!CreateMemTables(clientId)) {
        Utility::LogError("[client %u]: Create hal Memory table failed.", clientId);
        return;
    }
    auto memrecord = record.record.memoryRecord;
    if (memrecord.memType == MemOpType::MALLOC) {
        RecordMalloc(clientId, memrecord);
    } else if (memrecord.memType == MemOpType::FREE) {
        RecordFree(clientId, memrecord);
    }
    return;
}

void HalAnalyzer::CheckLeak(const size_t clientId)
{
    bool foundLeaks = false;
    for (const auto& pair :memtables_[clientId]) {
        if (pair.second != AddrStatus::FREE_ALREADY) {
            foundLeaks = true;
            Utility::LogWarn("[client %u]: Leak memory in Malloc operator, addr: 0x%lx", clientId, pair.first);
        }
    }
    if (!foundLeaks) {
        Utility::LogInfo("[client %u]: There is no leak memory.", clientId);
    }
}

void HalAnalyzer::LeakAnalyze()
{
    if (memtables_.empty()) {
        Utility::LogError("No memory records available.");
    } else {
        for (const auto& pair :memtables_) {
            CheckLeak(pair.first);
        }
    }

    return;
}

HalAnalyzer::~HalAnalyzer()
{
    LeakAnalyze();
}

}