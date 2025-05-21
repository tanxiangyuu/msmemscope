// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "memory_state_record.h"
#include "dump_record.h"
#include "utility/log.h"
#include "bit_field.h"
#include "module_info.h"

namespace Leaks {

MemoryStateRecord::MemoryStateRecord(Config config)
{
    config_ = config;
}

void MemoryStateRecord::RecordMemoryState(const Record& record, CallStackString& stack)
{
    std::lock_guard<std::mutex> lock(recordMutex_);
    auto type = record.eventRecord.type;
    auto it = memInfoProcessFuncMap_.find(type);
    if (it == memInfoProcessFuncMap_.end()) {
        return ;
    }
    it->second(record, stack);
}

void MemoryStateRecord::HostMemProcess(const MemOpRecord& memRecord, uint64_t& currentSize)
{
    if (memRecord.memType == MemOpType::MALLOC) {
        hostMemSizeMap_[memRecord.addr] = memRecord.memSize;
        currentSize = memRecord.memSize;
    } else if (hostMemSizeMap_.find(memRecord.addr) != hostMemSizeMap_.end()) {
        currentSize = hostMemSizeMap_[memRecord.addr];
        hostMemSizeMap_.erase(memRecord.addr);
    } else {
        return ;
    }
}

void MemoryStateRecord::HalMemProcess(MemOpRecord& memRecord, uint64_t& currentSize, std::string& deviceType)
{
    if (memRecord.memType == MemOpType::MALLOC) {
        memSizeMap_[memRecord.addr] = memRecord.memSize;
        currentSize = memRecord.memSize;
    } else {
        currentSize = memSizeMap_[memRecord.addr];
        memSizeMap_[memRecord.addr] = 0;
        // halfree目前device Id为N/A，需要和其他数据匹配
        auto key = std::make_pair("common", memRecord.addr);
        auto it = ptrMemoryInfoMap_.find(key);
        if (it == ptrMemoryInfoMap_.end() || ptrMemoryInfoMap_[key].size() == 0) {
            return ;
        }
        deviceType = ptrMemoryInfoMap_[key][0].container.deviceId;
    }
}

MemRecordAttr MemoryStateRecord::GetMemInfoAttr(MemOpRecord& memRecord, uint64_t currentSize)
{
    BitField<decltype(config_.analysisType)> analysisType(config_.analysisType);
    std::string halOwner = "";
    MemRecordAttr attr;
    if (memRecord.memType == MemOpType::MALLOC && memRecord.devType == DeviceType::NPU &&
        analysisType.checkBit(static_cast<size_t>(AnalysisType::DECOMPOSE_ANALYSIS))) {
        auto it = MODULE_HASH_TABLE.find(memRecord.modid);
        if (it != MODULE_HASH_TABLE.end()) {
            halOwner = "CANN@" + it->second;
        } else {
            halOwner = "CANN@UNKNOWN";
        }
        attr.owner = halOwner;
    }
    if (memRecord.devType == DeviceType::NPU) {
        attr.modid = memRecord.modid;
    }
    attr.addr = memRecord.addr;
    attr.size = currentSize;
    return attr;
}

void MemoryStateRecord::MemoryInfoProcess(const Record& record, CallStackString& stack)
{
    auto memRecord = record.eventRecord.record.memoryRecord;
    std::string memOp = memRecord.memType == MemOpType::MALLOC ? "MALLOC" : "FREE";
    auto ptr = memRecord.addr;
    uint64_t currentSize = 0;
    std::string deviceType = "";

    if (memRecord.devType == DeviceType::CPU) {
        HostMemProcess(memRecord, currentSize);
    } else {
        HalMemProcess(memRecord, currentSize, deviceType);
    }

    if (deviceType.empty()) {
        if (memRecord.devId == GD_INVALID_NUM) {
            deviceType = "N/A";
        } else {
            deviceType = memRecord.space == MemOpSpace::HOST || memRecord.devType == DeviceType::CPU ?
                    "host" : std::to_string(memRecord.devId);
        }
    }
    DumpContainer container;
    container.id = memRecord.recordIndex;
    container.event = memOp;
    container.eventType = "HAL";
    container.name = "N/A";
    container.timeStamp = memRecord.timeStamp;
    container.pid = memRecord.pid;
    container.tid = memRecord.tid;
    container.deviceId = deviceType;
    container.addr = std::to_string(memRecord.addr);

    MemRecordAttr attr = GetMemInfoAttr(memRecord, currentSize);
    std::ostringstream oss;
    if (memRecord.memType == MemOpType::FREE) {
        oss << "{addr:" << memRecord.addr << ",size:" << currentSize << ",MID:" << memRecord.modid << "}";
        std::string freeAttr = "\"" + oss.str() + "\"";
        container.attr = freeAttr;
    }

    auto key = std::make_pair("common", ptr);
    auto it = ptrMemoryInfoMap_.find(key);
    if (it == ptrMemoryInfoMap_.end()) {
        ptrMemoryInfoMap_.insert({key, {}});
    }
    MemStateInfo memInfo {};
    memInfo.container = container;
    memInfo.stack = stack;
    memInfo.attr = attr;
    ptrMemoryInfoMap_[key].push_back(memInfo);
}

inline void CopyMemPoolRecordMember(const MemPoolRecord &record, DumpContainer &container)
{
    container.id = record.recordIndex;
    container.pid = record.pid;
    container.tid = record.tid;
    container.timeStamp = record.timeStamp;
    container.deviceId = std::to_string(record.devId);
}

void MemoryStateRecord::PackDumpContainer(
    DumpContainer& container, const MemoryUsage& memoryUsage, const std::string memPoolType, MemRecordAttr& attr)
{
    std::string eventType = memoryUsage.allocSize >= 0 ? "MALLOC" : "FREE";
    attr.addr = memoryUsage.ptr;
    attr.size = memoryUsage.allocSize;

    BitField<decltype(config_.analysisType)> analysisType(config_.analysisType);
    if (eventType == "MALLOC" && analysisType.checkBit(static_cast<size_t>(AnalysisType::DECOMPOSE_ANALYSIS))) {
        attr.owner = memPoolType;
    }
    attr.totalReserved = memoryUsage.totalReserved;
    attr.totalAllocated = memoryUsage.totalAllocated;
    container.event = eventType;
    container.eventType = memPoolType;
    container.name = "N/A";
    container.addr = std::to_string(memoryUsage.ptr);
}

void MemoryStateRecord::MemoryPoolInfoProcess(const Record& record, CallStackString& stack)
{
    MemoryUsage memoryUsage { };
    std::string memPoolType { };
    DumpContainer container;
    memoryUsage = record.eventRecord.record.memPoolRecord.memoryUsage;
    CopyMemPoolRecordMember(record.eventRecord.record.memPoolRecord, container);
    switch (record.eventRecord.type) {
        case RecordType::TORCH_NPU_RECORD:
            memPoolType = "PTA";
            break;
        case RecordType::MINDSPORE_NPU_RECORD:
            memPoolType = "MINDSPORE";
            break;
        case RecordType::ATB_MEMORY_POOL_RECORD:
            memPoolType = "ATB";
            break;
        default:
            LOG_ERROR("Undefined memory pool type.");
            return;
    }
    MemRecordAttr attr;
    PackDumpContainer(container, memoryUsage, memPoolType, attr);

    std::ostringstream oss;
    if (memoryUsage.allocSize < 0) {
        oss << "{addr:" << memoryUsage.ptr << ",size:" << memoryUsage.allocSize <<
            ",total:" << memoryUsage.totalReserved << ",used:" << memoryUsage.totalAllocated << "}";
        std::string freeAttr = "\"" + oss.str() + "\"";
        container.attr = freeAttr;
    }

    auto key = std::make_pair(memPoolType, memoryUsage.ptr);
    auto it = ptrMemoryInfoMap_.find(key);
    if (it == ptrMemoryInfoMap_.end()) {
        ptrMemoryInfoMap_.insert({key, {}});
    }
    MemStateInfo memInfo {};
    memInfo.container = container;
    memInfo.stack = stack;
    memInfo.attr = attr;
    ptrMemoryInfoMap_[key].push_back(memInfo);
}

void MemoryStateRecord::MemoryAccessInfoProcess(const Record& record, CallStackString& stack)
{
    auto memAccessRecord = record.eventRecord.record.memAccessRecord;
    std::string eventType;
    auto ptr = memAccessRecord.addr;
    switch (memAccessRecord.eventType) {
        case Leaks::AccessType::READ: {
            eventType = "READ";
            break;
        }
        case Leaks::AccessType::WRITE: {
            eventType = "WRITE";
            break;
        }
        default: {
            eventType = "UNKNOWN";
            break;
        }
    }

    std::ostringstream oss;
    oss << "\"{addr:" << memAccessRecord.addr << ",size:"
        << memAccessRecord.memSize << "," << memAccessRecord.attr << "}\"";
    std::string attr = oss.str();

    DumpContainer container;
    container.id = memAccessRecord.recordIndex;
    container.event = "ACCESS";
    container.eventType = eventType;
    container.name = memAccessRecord.name;
    container.timeStamp = memAccessRecord.timestamp;
    container.pid = memAccessRecord.pid;
    container.tid = memAccessRecord.tid;
    container.deviceId = std::to_string(memAccessRecord.devId);
    container.addr = std::to_string(memAccessRecord.addr);
    container.attr = attr;

    auto key = memAccessRecord.memType == AccessMemType::ATEN?
        std::make_pair("PTA", ptr) : std::make_pair("ATB", ptr);
    auto it = ptrMemoryInfoMap_.find(key);
    if (it == ptrMemoryInfoMap_.end()) {
        ptrMemoryInfoMap_.insert({key, {}});
    }
    MemStateInfo memInfo {};
    memInfo.container = container;
    memInfo.stack = stack;
    ptrMemoryInfoMap_[key].push_back(memInfo);
}

const std::vector<MemStateInfo>& MemoryStateRecord::GetPtrMemInfoList(std::pair<std::string, int64_t> key)
{
    auto it = ptrMemoryInfoMap_.find(key);
    if (it == ptrMemoryInfoMap_.end()) {
        ptrMemoryInfoMap_.insert({key, {}});
    }
    return ptrMemoryInfoMap_[key];
}

void MemoryStateRecord::SetPtrMemInfoList(std::pair<std::string, int64_t> key, std::vector<MemStateInfo>& infoList)
{
    auto it = ptrMemoryInfoMap_.find(key);
    if (it == ptrMemoryInfoMap_.end()) {
        return ;
    }
    ptrMemoryInfoMap_[key] = infoList;
}

void MemoryStateRecord::DeleteMemStateInfo(std::pair<std::string, uint64_t> key)
{
    auto it = ptrMemoryInfoMap_.find(key);
    if (it != ptrMemoryInfoMap_.end()) {
        ptrMemoryInfoMap_.erase(key);
    }
}

}
