// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "memory_state_record.h"
#include "dump_record.h"

namespace Leaks {

std::map<int32_t, std::string> g_moduleIdComponetsMap = {
    {0, "SLOG"}, {1, "IDEDD"}, {2, "IDEDH"}, {3, "HCCL"}, {4, "FMK"}, {5, "HIAIENGINE"},
    {6, "DVPP"}, {7, "RUNTIME"}, {8, "CCE"}, {9, "HDC"}, {10, "DRV"}, {11, "MDCFUSION"},
    {12, "MDCLOCATION"}, {13, "MDCPERCEPTION"}, {14, "MDCFSM"}, {15, "MDCCOMMON"}, {16, "MDCMONITOR"},
    {17, "MDCBSWP"}, {18, "MDCDEFAULT"}, {19, "MDCSC"}, {20, "MDCPNC"}, {21, "MLL"}, {22, "DEVMM"},
    {23, "KERNEL"}, {24, "LIBMEDIA"}, {25, "CCECPU"}, {26, "ASCENDDK"}, {27, "ROS"}, {28, "HCCP"},
    {29, "ROCE"}, {30, "TEFUSION"}, {31, "PROFILING"}, {32, "DP"}, {33, "APP"}, {34, "TS"}, {35, "TSDUMP"},
    {36, "AICPU"}, {37, "LP"}, {38, "TDT"}, {39, "FE"}, {40, "MD"}, {41, "MB"}, {42, "ME"}, {43, "IMU"},
    {44, "IMP"}, {45, "GE"}, {46, "MDCFUSA"}, {47, "CAMERA"}, {48, "ASCENDCL"}, {49, "TEEOS"}, {50, "ISP"},
    {51, "SIS"}, {52, "HSM"}, {53, "DSS"}, {54, "PROCMGR"}, {55, "BBOX"}, {56, "AIVECTOR"}, {57, "TBE"},
    {58, "FV"}, {59, "MDCMAP"}, {60, "TUNE"}, {61, "HSS"}, {62, "FFTS"}, {63, "OP"}, {64, "UDF"},
    {65, "HICAID"}, {66, "TSYNC"}, {67, "AUDIO"}, {68, "TPRT"}, {69, "ASCENDCKERNEL"}, {70, "ASYS"},
    {71, "ATRACE"}, {72, "RTC"}, {73, "SYSMONITOR"}, {74, "AML"}, {255, "UNKNOWN"}
};

MemoryStateRecord::MemoryStateRecord(Config config)
{
    config_ = config;
}

std::map<std::pair<std::string, uint64_t>, std::vector<MemStateInfo>>& MemoryStateRecord::GetPtrMemoryInfoMap()
{
    return ptrMemoryInfoMap_;
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

void MemoryStateRecord::SaveMemInfoData(std::pair<std::string, uint64_t> key, DumpContainer& container,
    CallStackString& stack)
{
    auto it = ptrMemoryInfoMap_.find(key);
    if (it == ptrMemoryInfoMap_.end()) {
        ptrMemoryInfoMap_.insert({key, {}});
    }
    MemStateInfo memInfo {};
    memInfo.container = container;
    memInfo.stack = stack;
    ptrMemoryInfoMap_[key].push_back(memInfo);
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

void MemoryStateRecord::GetHalComponet(MemOpRecord& memRecord, std::string& halOwner)
{
    BitField<decltype(config_.analysisType)> analysisType(config_.analysisType);
    // halfree目前moduleId为-1，需要和malloc事件匹配对应的component
    if (memRecord.memType == MemOpType::FREE) {
        auto key = std::make_pair("common", memRecord.addr);
        auto it = ptrMemoryInfoMap_.find(key);
        if (it == ptrMemoryInfoMap_.end() || ptrMemoryInfoMap_[key][0].container.event != "MALLOC") {
            return ;
        }
        halOwner = Utility::ExtractAttrValueByKey(ptrMemoryInfoMap_[key][0].container.attr, "owner");
        return ;
    }

    if (analysisType.checkBit(static_cast<size_t>(AnalysisType::DECOMPOSE_ANALYSIS))) {
        auto it = g_moduleIdComponetsMap.find(memRecord.modid);
        if (it != g_moduleIdComponetsMap.end()) {
            halOwner = "CANN@{" + g_moduleIdComponetsMap[memRecord.modid] + "}";
        } else {
            halOwner = "CANN@{UNKNOWN}";
        }
    }
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

    std::string halOwner = "";
    GetHalComponet(memRecord, halOwner);
    std::ostringstream oss;
    oss << "{addr:" << memRecord.addr << ",size:" << currentSize << ",owner:" << halOwner <<
        ",MID:" << memRecord.modid << "}";
    std::string attr = "\"" + oss.str() + "\"";

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
    container.attr = attr;

    auto key = std::make_pair("common", ptr);
    SaveMemInfoData(key, container, stack);
}

void MemoryStateRecord::MemoryPoolInfoProcess(const Record& record, CallStackString& stack)
{
    MemoryUsage memoryUsage { };
    std::string memPoolType { };
    DumpContainer container;
    if (record.eventRecord.type == RecordType::TORCH_NPU_RECORD) {
        memoryUsage = record.eventRecord.record.torchNpuRecord.memoryUsage;
        memPoolType = "PTA";
        CopyMemPoolRecordMember(record.eventRecord.record.torchNpuRecord, container);
    } else if (record.eventRecord.type == RecordType::MINDSPORE_NPU_RECORD) {
        memoryUsage = record.eventRecord.record.mindsporeNpuRecord.memoryUsage;
        memPoolType = "Mindspore";
        CopyMemPoolRecordMember(record.eventRecord.record.mindsporeNpuRecord, container);
    } else {
        memoryUsage = record.eventRecord.record.atbMemPoolRecord.memoryUsage;
        memPoolType = "ATB";
        CopyMemPoolRecordMember(record.eventRecord.record.atbMemPoolRecord, container);
    }

    std::string eventType = memoryUsage.allocSize >= 0 ? "MALLOC" : "FREE";
    auto ptr = memoryUsage.ptr;

    std::ostringstream oss;
    oss << "{addr:" << ptr << ",size:" << memoryUsage.allocSize << ",owner:" << ",total:" <<
        memoryUsage.totalReserved << ",used:" << memoryUsage.totalAllocated << "}";
    std::string attr = "\"" + oss.str() + "\"";

    container.event = eventType;
    container.eventType = memPoolType;
    container.name = "N/A";
    container.addr = std::to_string(memoryUsage.ptr);
    container.attr = attr;

    auto key = std::make_pair(memPoolType, ptr);
    SaveMemInfoData(key, container, stack);
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
    SaveMemInfoData(key, container, stack);
}

const std::vector<MemStateInfo>& MemoryStateRecord::GetPtrMemInfoList(std::pair<std::string, int64_t> key)
{
    auto it = ptrMemoryInfoMap_.find(key);
    if (it == ptrMemoryInfoMap_.end()) {
        ptrMemoryInfoMap_.insert({key, {}});
    }
    return ptrMemoryInfoMap_[key];
}

void MemoryStateRecord::DeleteMemStateInfo(std::pair<std::string, uint64_t> key)
{
    auto it = ptrMemoryInfoMap_.find(key);
    if (it != ptrMemoryInfoMap_.end()) {
        ptrMemoryInfoMap_.erase(key);
    }
}

MemoryStateRecord::~MemoryStateRecord()
{
    std::lock_guard<std::mutex> lock(recordMutex_);
    for (auto it = ptrMemoryInfoMap_.begin(); it != ptrMemoryInfoMap_.end();) {
        auto key = it->first;
        auto memInfoLists = it->second;
        for (auto memInfo : memInfoLists) {
            DumpRecord::GetInstance(config_).WriteToFile(memInfo.container, memInfo.stack);
        }
        DeleteMemStateInfo(key);
        ++it;
    }
}

}
