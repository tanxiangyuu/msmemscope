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

void MemoryStateRecord::RecordMemoryState(const RecordBase& record)
{
    std::lock_guard<std::mutex> lock(recordMutex_);
    auto type = record.type;
    auto it = memInfoProcessFuncMapV2_.find(type);
    if (it == memInfoProcessFuncMapV2_.end()) {
        return;
    }
    it->second(record);
}

void MemoryStateRecord::HostMemProcess(const MemOpRecord& memRecord, uint64_t& currentSize)
{
    if (memRecord.subtype == RecordSubType::MALLOC) {
        hostMemSizeMap_[memRecord.addr] = memRecord.memSize;
        currentSize = memRecord.memSize;
    } else if (hostMemSizeMap_.find(memRecord.addr) != hostMemSizeMap_.end()) {
        currentSize = hostMemSizeMap_[memRecord.addr];
        hostMemSizeMap_.erase(memRecord.addr);
    } else {
        LOG_DEBUG("No matching malloc operation found for free operator: addr: 0x%lx", memRecord.addr);
        return ;
    }
}

void MemoryStateRecord::HalMemProcess(const MemOpRecord& memRecord, uint64_t& currentSize, std::string& deviceType)
{
    if (memRecord.subtype == RecordSubType::MALLOC) {
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

MemRecordAttr MemoryStateRecord::GetMemInfoAttr(const MemOpRecord& memRecord, uint64_t currentSize)
{
    BitField<decltype(config_.analysisType)> analysisType(config_.analysisType);
    std::string halOwner = "";
    MemRecordAttr attr;
    if (memRecord.subtype == RecordSubType::MALLOC && memRecord.devType == DeviceType::NPU &&
        analysisType.checkBit(static_cast<size_t>(AnalysisType::DECOMPOSE_ANALYSIS))) {
        auto it = MODULE_HASH_TABLE.find(memRecord.modid);
        if (it != MODULE_HASH_TABLE.end()) {
            halOwner = "CANN@" + it->second;
        } else {
            halOwner = "CANN@UNKNOWN";
        }
        attr.leaksDefinedOwner = halOwner;
        const TLVBlock* tlv = GetTlvBlock(memRecord, TLVBlockType::MEM_OWNER);
        attr.userDefinedOwner = std::string(tlv == nullptr ? "" : tlv->data);
    }
    if (memRecord.devType == DeviceType::NPU) {
        attr.modid = memRecord.modid;
    }
    attr.addr = memRecord.addr;
    attr.size = currentSize;
    return attr;
}

void MemoryStateRecord::MemoryInfoProcess(const RecordBase& record)
{
    const MemOpRecord& memRecord = static_cast<const MemOpRecord&>(record);
    std::string memOp = memRecord.subtype == RecordSubType::MALLOC ? "MALLOC" : "FREE";
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
    container.timestamp = memRecord.timestamp;
    container.pid = memRecord.pid;
    container.tid = memRecord.tid;
    container.deviceId = deviceType;
    container.addr = std::to_string(memRecord.addr);

    MemRecordAttr attr = GetMemInfoAttr(memRecord, currentSize);
    std::ostringstream oss;
    if (memRecord.subtype == RecordSubType::FREE) {
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

    const TLVBlock* cstack = GetTlvBlock(memRecord, TLVBlockType::CALL_STACK_C);
    const TLVBlock* pystack = GetTlvBlock(memRecord, TLVBlockType::CALL_STACK_PYTHON);
    memInfo.stack.cStack = cstack == nullptr ? std::string("") : std::string(cstack->data);
    memInfo.stack.pyStack = pystack == nullptr ? std::string("") : std::string(pystack->data);
    memInfo.attr = attr;
    ptrMemoryInfoMap_[key].push_back(memInfo);
}

inline void CopyMemPoolRecordMember(const MemPoolRecord &record, DumpContainer &container)
{
    container.id = record.recordIndex;
    container.pid = record.pid;
    container.tid = record.tid;
    container.timestamp = record.timestamp;
    container.deviceId = std::to_string(record.devId);
    container.owner += std::string(record.owner);
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
    PackDumpContainer(container, record.eventRecord.record.memPoolRecord, memPoolType, attr);

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

void MemoryStateRecord::MemoryAddrInfoProcess(const Record& record, CallStackString& stack)
{
    BitField<decltype(config_.analysisType)> analysisType(config_.analysisType);
    if (!analysisType.checkBit(static_cast<size_t>(AnalysisType::DECOMPOSE_ANALYSIS))) {
        return;
    }

    auto addrInfoRecord = record.eventRecord.record.addrInfo;
    auto key = std::make_pair("PTA", addrInfoRecord.addr);
    if (!ptrMemoryInfoMap_.count(key)) {
        return;
    }

    auto& addrInfo = ptrMemoryInfoMap_[key][0];
    if (addrInfo.container.event != "MALLOC") {
        return;
    }

    std::string owner = std::string(addrInfoRecord.owner);
    if (addrInfoRecord.type == AddrInfoType::USER_DEFINED) {
        addrInfo.attr.userDefinedOwner += owner;
    } else if (addrInfoRecord.type == AddrInfoType::PTA_OPTIMIZER_STEP) {
        UpdateLeaksDefinedOwner(addrInfo.attr.leaksDefinedOwner, owner);
    }
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
    PackDumpContainer(container, memAccessRecord, eventType, attr);

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

    if (memAccessRecord.memType == AccessMemType::ATEN && ptrMemoryInfoMap_[key][0].container.event == "MALLOC") {
        UpdateLeaksDefinedOwner(ptrMemoryInfoMap_[key][0].attr.leaksDefinedOwner, "@ops@aten");
    }
}

const std::vector<MemStateInfo>& MemoryStateRecord::GetPtrMemInfoList(std::pair<std::string, int64_t> key)
{
    auto it = ptrMemoryInfoMap_.find(key);
    if (it == ptrMemoryInfoMap_.end()) {
        ptrMemoryInfoMap_.insert({key, {}});
    }
    return ptrMemoryInfoMap_[key];
}

std::map<std::pair<std::string, uint64_t>, std::vector<MemStateInfo>>& MemoryStateRecord::GetPtrMemInfoMap()
{
    return ptrMemoryInfoMap_;
}

void MemoryStateRecord::DeleteMemStateInfo(std::pair<std::string, uint64_t> key)
{
    auto it = ptrMemoryInfoMap_.find(key);
    if (it != ptrMemoryInfoMap_.end()) {
        ptrMemoryInfoMap_.erase(key);
    }
}

void MemoryStateRecord::PackDumpContainer(
    DumpContainer& container, const MemPoolRecord& memPool, const std::string& memPoolType, MemRecordAttr& attr)
{
    std::string eventType = memPool.memoryUsage.allocSize >= 0 ? "MALLOC" : "FREE";
    attr.addr = memPool.memoryUsage.ptr;
    attr.size = memPool.memoryUsage.allocSize;

    BitField<decltype(config_.analysisType)> analysisType(config_.analysisType);
    if (eventType == "MALLOC" && analysisType.checkBit(static_cast<size_t>(AnalysisType::DECOMPOSE_ANALYSIS))) {
        attr.leaksDefinedOwner = memPoolType;
        attr.userDefinedOwner = std::string(memPool.owner);
    }
    attr.totalReserved = memPool.memoryUsage.totalReserved;
    attr.totalAllocated = memPool.memoryUsage.totalAllocated;
    container.event = eventType;
    container.eventType = memPoolType;
    container.name = "N/A";
    container.addr = std::to_string(memPool.memoryUsage.ptr);
}

void MemoryStateRecord::PackDumpContainer(
    DumpContainer& container, const MemAccessRecord& memAccessRecord,
    const std::string& eventType, const std::string& attr)
{
    container.id = memAccessRecord.recordIndex;
    container.event = "ACCESS";
    container.eventType = eventType;
    container.name = memAccessRecord.name;
    container.timestamp = memAccessRecord.timestamp;
    container.pid = memAccessRecord.pid;
    container.tid = memAccessRecord.tid;
    container.deviceId = std::to_string(memAccessRecord.devId);
    container.addr = std::to_string(memAccessRecord.addr);
    container.attr = attr;
}

void MemoryStateRecord::UpdateLeaksDefinedOwner(std::string& owner, const std::string& newOwner)
{
    static std::string ptaPrefix = "PTA";
    static size_t ptaPrefixLen = ptaPrefix.length();

    if (owner.rfind(ptaPrefix, 0) != 0) {
        return;
    }

    if (owner.length() == ptaPrefixLen) {
        owner += newOwner;
    } else {
        // 部分内存有可能先作为算子操作的内容，然后被识别为其他类型，如weight，
        // 则优先用weight覆盖aten，而aten不能覆盖其他类型
        if (newOwner != "@ops@aten") {
            owner = ptaPrefix + newOwner;
        }
    }
}

}
