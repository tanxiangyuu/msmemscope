// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "memory_state_record.h"
#include "dump_record.h"
#include "utility/log.h"
#include "bit_field.h"
#include "module_info.h"
#include "data_handler.h"

namespace Leaks {

MemoryStateRecord::MemoryStateRecord(Config config)
{
    config_ = config;
}

void MemoryStateRecord::RecordMemoryState(const RecordBase& record)
{
    auto type = record.type;
    auto it = memInfoProcessFuncMapV2_.find(type);
    if (it == memInfoProcessFuncMapV2_.end()) {
        return;
    }
    it->second(record);
}

void MemoryStateRecord::HostMemProcess(const MemOpRecord& memRecord, uint64_t& currentSize)
{
    std::lock_guard<std::mutex> lock(memSizeMutex_);
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
    std::lock_guard<std::mutex> lock(memSizeMutex_);
    if (memRecord.subtype == RecordSubType::MALLOC) {
        memSizeMap_[memRecord.addr] = memRecord.memSize;
        currentSize = memRecord.memSize;
    } else {
        currentSize = memSizeMap_[memRecord.addr];
        memSizeMap_[memRecord.addr] = 0;
        // halfree目前device Id为N/A，需要和其他数据匹配
        auto key = std::make_pair("common", memRecord.addr);
        if (ptrMemoryInfoMap_.find(key) == ptrMemoryInfoMap_.end() || ptrMemoryInfoMap_[key].size() == 0) {
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
    std::lock_guard<std::mutex> lock(memInfoMutex_);
    if (ptrMemoryInfoMap_.find(key) == ptrMemoryInfoMap_.end()) {
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
    const TLVBlock* owner = GetTlvBlock(record, TLVBlockType::ADDR_OWNER);
    container.id = record.recordIndex;
    container.pid = record.pid;
    container.tid = record.tid;
    container.timestamp = record.timestamp;
    container.deviceId = std::to_string(record.devId);
    container.owner += owner == nullptr ? std::string("") : std::string(owner->data);
}

void MemoryStateRecord::MemoryPoolInfoProcess(const RecordBase& record)
{
    const MemPoolRecord& memPoolRecord = static_cast<const MemPoolRecord&>(record);
    MemoryUsage memoryUsage { };
    std::string memPoolType { };
    DumpContainer container;
    memoryUsage = memPoolRecord.memoryUsage;
    CopyMemPoolRecordMember(memPoolRecord, container);
    switch (memPoolRecord.type) {
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
    PackDumpContainer(container, memPoolRecord, memPoolType, attr);

    std::ostringstream oss;

    // 内存事件为free事件
    if (memoryUsage.dataType == 1) {
        oss << "{addr:" << memoryUsage.ptr << ",size:" << memoryUsage.allocSize <<
            ",total:" << memoryUsage.totalReserved << ",used:" << memoryUsage.totalAllocated << "}";
        std::string freeAttr = "\"" + oss.str() + "\"";
        container.attr = freeAttr;
    }
    auto key = std::make_pair(memPoolType, memoryUsage.ptr);
    std::lock_guard<std::mutex> lock(memInfoMutex_);
    if (ptrMemoryInfoMap_.find(key) == ptrMemoryInfoMap_.end()) {
        ptrMemoryInfoMap_.insert({key, {}});
    }
    MemStateInfo memInfo {};
    memInfo.container = container;
    const TLVBlock* cstack = GetTlvBlock(memPoolRecord, TLVBlockType::CALL_STACK_C);
    const TLVBlock* pystack = GetTlvBlock(memPoolRecord, TLVBlockType::CALL_STACK_PYTHON);
    memInfo.stack.cStack = cstack == nullptr ? std::string("") : std::string(cstack->data);
    memInfo.stack.pyStack = pystack == nullptr ? std::string("") : std::string(pystack->data);
    memInfo.attr = attr;
    ptrMemoryInfoMap_[key].push_back(memInfo);
}

void MemoryStateRecord::MemoryAddrInfoProcess(const RecordBase& record)
{
    BitField<decltype(config_.analysisType)> analysisType(config_.analysisType);
    if (!analysisType.checkBit(static_cast<size_t>(AnalysisType::DECOMPOSE_ANALYSIS))) {
        return;
    }
    const AddrInfo& addrInfoRecord = static_cast<const AddrInfo&>(record);
    auto key = std::make_pair("PTA", addrInfoRecord.addr);
    if (!ptrMemoryInfoMap_.count(key)) {
        return;
    }

    auto& addrInfo = ptrMemoryInfoMap_[key][0];
    if (addrInfo.container.event != "MALLOC") {
        return;
    }
    const TLVBlock* ownerTlv = GetTlvBlock(addrInfoRecord, TLVBlockType::ADDR_OWNER);
    std::string owner = ownerTlv == nullptr ? "N/A" : ownerTlv->data;

    if (addrInfoRecord.addrInfoType == AddrInfoType::USER_DEFINED) {
        addrInfo.attr.userDefinedOwner += owner;
    } else if (addrInfoRecord.addrInfoType == AddrInfoType::PTA_OPTIMIZER_STEP) {
        UpdateLeaksDefinedOwner(addrInfo.attr.leaksDefinedOwner, owner);
    }
}

void MemoryStateRecord::MemoryAccessInfoProcess(const RecordBase& record)
{
    const MemAccessRecord& memAccessRecord = static_cast<const MemAccessRecord&>(record);
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
    const TLVBlock* attrTlv = GetTlvBlock(memAccessRecord, TLVBlockType::MEM_ATTR);
    std::string attrData = attrTlv == nullptr ? "N/A" : attrTlv->data;
    std::ostringstream oss;
    oss << "\"{addr:" << memAccessRecord.addr << ",size:"
        << memAccessRecord.memSize << "," << attrData << "}\"";
    std::string attr = oss.str();

    DumpContainer container;
    PackDumpContainer(container, memAccessRecord, eventType, attr);

    auto key = memAccessRecord.memType == AccessMemType::ATEN?
        std::make_pair("PTA", ptr) : std::make_pair("ATB", ptr);
    {
        std::lock_guard<std::mutex> lock(memInfoMutex_);
        if (ptrMemoryInfoMap_.find(key) == ptrMemoryInfoMap_.end()) {
            ptrMemoryInfoMap_.insert({key, {}});
        }
        MemStateInfo memInfo {};
        memInfo.container = container;
        const TLVBlock* cstack = GetTlvBlock(memAccessRecord, TLVBlockType::CALL_STACK_C);
        const TLVBlock* pystack = GetTlvBlock(memAccessRecord, TLVBlockType::CALL_STACK_PYTHON);
        memInfo.stack.cStack = cstack == nullptr ? std::string("") : std::string(cstack->data);
        memInfo.stack.pyStack = pystack == nullptr ? std::string("") : std::string(pystack->data);
        ptrMemoryInfoMap_[key].push_back(memInfo);
    }

    if (memAccessRecord.memType == AccessMemType::ATEN && ptrMemoryInfoMap_[key][0].container.event == "MALLOC") {
        UpdateLeaksDefinedOwner(ptrMemoryInfoMap_[key][0].attr.leaksDefinedOwner, "@ops@aten");
    }
}

const std::vector<MemStateInfo>& MemoryStateRecord::GetPtrMemInfoList(std::pair<std::string, int64_t> key)
{
    std::lock_guard<std::mutex> lock(memInfoMutex_);
    if (ptrMemoryInfoMap_.find(key) == ptrMemoryInfoMap_.end()) {
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
    std::lock_guard<std::mutex> lock(memInfoMutex_);
    if (ptrMemoryInfoMap_.find(key) != ptrMemoryInfoMap_.end()) {
        ptrMemoryInfoMap_.erase(key);
    }
}

void MemoryStateRecord::PackDumpContainer(
    DumpContainer& container, const MemPoolRecord& memPool, const std::string& memPoolType, MemRecordAttr& attr)
{
    const TLVBlock* owner = GetTlvBlock(memPool, TLVBlockType::ADDR_OWNER);
    std::string eventType = memPool.memoryUsage.dataType == 0 ? "MALLOC" : "FREE";
    attr.addr = memPool.memoryUsage.ptr;
    attr.size = memPool.memoryUsage.allocSize;

    BitField<decltype(config_.analysisType)> analysisType(config_.analysisType);
    if (eventType == "MALLOC" && analysisType.checkBit(static_cast<size_t>(AnalysisType::DECOMPOSE_ANALYSIS))) {
        attr.leaksDefinedOwner = memPoolType;
        attr.userDefinedOwner = owner == nullptr ? std::string("") : std::string(owner->data);
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
    const TLVBlock* nameTlv = GetTlvBlock(memAccessRecord, TLVBlockType::OP_NAME);
    std::string name = nameTlv == nullptr ? "N/A" : nameTlv->data;
    container.id = memAccessRecord.recordIndex;
    container.event = "ACCESS";
    container.eventType = eventType;
    container.name = name;
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
