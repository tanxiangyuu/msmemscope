// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "hal_analyzer.h"
#include "utility/log.h"

namespace Leaks {

HalAnalyzer& HalAnalyzer::GetInstance()
{
    static HalAnalyzer analyzer;
    return analyzer;
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
        Utility::LogWarn("[client %u][device: %ld]: Malloc operator did not find %d Module in index %u malloc record.",
            clientId, memrecord.devId, memrecord.modid, memrecord.recordIndex);
    }

    Utility::LogInfo(
        "[client %u][device: %ld]: server malloc record, index: %u, addr: 0x%lx, size: %u, space: %u, module: %s",
        clientId, memrecord.devId, memrecord.recordIndex,
        memrecord.addr, memrecord.memSize, static_cast<uint8_t>(memrecord.space), modulename.c_str());

    if (memtables_[clientId].find(memkey) != memtables_[clientId].end() &&
        (memtables_[clientId].find(memkey)->second.addrStatus == AddrStatus::FREE_WAIT)) {
        Utility::LogError("[client %u]: server already has malloc record in addr: 0x%lx ,", clientId, memrecord.addr);
        Utility::LogError("[client %u]: but now malloc again in index: %u, addr: 0x%lx, size: %u, space: %u",
            clientId, memrecord.recordIndex, memrecord.addr, memrecord.memSize, memrecord.space);
    }
    memtables_[clientId][memkey].deviceId = memrecord.devId;
    memtables_[clientId][memkey].addrStatus = AddrStatus::FREE_WAIT;
}

void HalAnalyzer::RecordFree(const ClientId &clientId, const MemOpRecord memrecord)
{
    uint64_t memkey = memrecord.addr;
    auto it = memtables_[clientId].find(memkey);
    int32_t freeDevId = GD_INVALID_NUM;
    if (it != memtables_[clientId].end()) {
        freeDevId = memtables_[clientId][memkey].deviceId;
        if (it->second.addrStatus == AddrStatus::FREE_WAIT) {
            memtables_[clientId][memkey].addrStatus = AddrStatus::FREE_ALREADY;
        } else {
            Utility::LogError("[client %u]: Double free operator found for malloc operation : addr: 0x%lx",
                clientId, memrecord.addr);
        }
    } else {
            Utility::LogError("[client %u]: No matching malloc operation found for free operator: addr: 0x%lx",
                clientId, memrecord.addr);
    }
    Utility::LogInfo("[client %u][device: %ld]: server free record, index: %u, addr: 0x%lx",
        clientId, freeDevId, memrecord.recordIndex, memrecord.addr);
}

bool HalAnalyzer::Record(const ClientId &clientId, const EventRecord &record)
{
    // 目前不处理CPU侧数据
    if (record.record.memoryRecord.devType == DeviceType::CPU) {
        return true;
    }
    if (!CreateMemTables(clientId)) {
        Utility::LogError("[client %u]: Create hal Memory table failed.", clientId);
        return false;
    }
    auto memrecord = record.record.memoryRecord;
    if (memrecord.memType == MemOpType::MALLOC) {
        RecordMalloc(clientId, memrecord);
        return true;
    } else if (memrecord.memType == MemOpType::FREE) {
        RecordFree(clientId, memrecord);
        return true;
    }
    return false;
}

void HalAnalyzer::CheckLeak(const size_t clientId)
{
    bool foundLeaks = false;
    for (const auto& pair :memtables_[clientId]) {
        if (pair.second.addrStatus != AddrStatus::FREE_ALREADY) {
            foundLeaks = true;
            Utility::LogWarn("[client %u]: Leak memory in Malloc operator, addr: 0x%lx", clientId, pair.first);
        }
    }
    if (!foundLeaks) {
        Utility::LogInfo("[client %u]: There is no hal leak memory.", clientId);
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