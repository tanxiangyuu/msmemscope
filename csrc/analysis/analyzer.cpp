// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "analyzer.h"
#include "log.h"
#include "module_info.h"

namespace Leaks {

constexpr uint64_t MEM_MODULE_ID_BIT = 56;

inline int32_t GetMallocModuleId(unsigned long long flag)
{
    // bit56~63: model id
    return (flag & 0xFF00000000000000) >> MEM_MODULE_ID_BIT;
}

void Analyzer::RecordMalloc(const ClientId &clientId, const MemOpRecord memrecord)
{
    uint64_t memkey = memrecord.addr;
    // malloc操作需解析当前moduleId
    auto flag = memrecord.flag;
    int32_t flagId = GetMallocModuleId(flag);
    bool foundModule = false;
    std::string modulename = "INVLID_MOUDLE_ID";
    if (g_ModuleHashTable.find(flagId) != g_ModuleHashTable.end()) {
        modulename = g_ModuleHashTable.find(flagId)->second;
        foundModule = true;
    }
    if (!foundModule) {
        Utility::LogError("[client %u]: Malloc operator did not find %d Module in index %u malloc record.",
            clientId, flagId, memrecord.recordIndex);
    }

    Utility::LogInfo("[client %u]: server malloc record, index: %u, addr: 0x%lx, size: %u, space: %u, module: %s",
        clientId, memrecord.recordIndex, memrecord.addr, memrecord.memSize, memrecord.space, modulename.c_str());

    if (memtables_[clientId].find(memkey) != memtables_[clientId].end() &&
        (memtables_[clientId].find(memkey)->second == AddrStatus::FREE_WAIT)) {
        Utility::LogError("[client %u]: server already has malloc record in addr: 0x%lx ,",
            " but now malloc again in index: %u, addr: 0x%lx, size: %u, space: %u",
            clientId, memrecord.addr,  memrecord.recordIndex, memrecord.addr, memrecord.memSize, memrecord.space);
    }
    memtables_[clientId][memkey] = AddrStatus::FREE_WAIT;
}

void Analyzer::RecordFree(const ClientId &clientId, const MemOpRecord memrecord)
{
    uint64_t memkey = memrecord.addr;
    Utility::LogInfo("[client %u]: server free record, index: %u, addr: 0x%lx",
        clientId, memrecord.recordIndex, memrecord.addr);

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

void Analyzer::Record(const ClientId &clientId, const EventRecord &record)
{
    CreateMemTables(clientId);
    auto memrecord = record.record.memoryRecord;
    if (memrecord.memType == MemOpType::MALLOC) {
        RecordMalloc(clientId, memrecord);
    } else if (memrecord.memType == MemOpType::FREE) {
        RecordFree(clientId, memrecord);
    }
    return;
}

void Analyzer::CheckLeak(const size_t clientId)
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

Analyzer::Analyzer(const AnalysisConfig &config)
{
    config_ = config;
}

void Analyzer::CreateMemTables(const ClientId &clientId)
{
    if (memtables_.find(clientId) != memtables_.end()) {
        return;
    }   else {
        Utility::LogInfo("[client %u]: Start Record Memory.", clientId);
        MemoryRecordTable memrecordtable{};
        memtables_[clientId] = memrecordtable;
        return;
    }
}

void Analyzer::Do(const ClientId &clientId, const EventRecord &record)
{
    switch (record.type) {
        case RecordType::MEMORY_RECORD: {
            Record(clientId, record);
            break;
        }
        case RecordType::KERNEL_LAUNCH_RECORD: {
            auto kernelLaunchRecord = record.record.kernelLaunchRecord;
            Utility::LogInfo("server kernelLaunch record, index: %u, type: %u, time: %u",
                kernelLaunchRecord.recordIndex,
                kernelLaunchRecord.type,
                kernelLaunchRecord.timeStamp);
            break;
        }
        case RecordType::ACL_ITF_RECORD: {
            auto aclItfRecord = record.record.aclItfRecord;
            Utility::LogInfo("server aclItf record, index: %u, type: %u, time: %u",
                aclItfRecord.recordIndex,
                aclItfRecord.type,
                aclItfRecord.timeStamp);
            break;
        }
        default:
            break;
    }

    return;
}

void Analyzer::LeakAnalyze()
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

Analyzer::~Analyzer()
{
    LeakAnalyze();
}

}
