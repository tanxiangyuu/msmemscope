// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef MEMORY_STATE_RECORD_H
#define MEMORY_STATE_RECORD_H

#include <functional>
#include <sstream>
#include <map>
#include <mutex>

#include "record_info.h"
#include "config_info.h"
#include "bit_field.h"

namespace Leaks {

using HandlerFunc = std::function<void(const Record&, CallStackString&)>;

class MemRecordAttr {
public:
    uint64_t addr;
    uint64_t size;
    std::string owner;
    int32_t modid;
    int64_t totalAllocated;
    int64_t totalReserved;
};

class MemStateInfo {
public:
    DumpContainer container;
    CallStackString stack;
    MemRecordAttr attr;
};

class MemoryStateRecord {
public:
    void RecordMemoryState(const Record& record, CallStackString& stack);
    void MemoryInfoProcess(const Record& record, CallStackString& stack);
    void MemoryPoolInfoProcess(const Record& record, CallStackString& stack);
    void MemoryAccessInfoProcess(const Record& record, CallStackString& stack);
    void MemoryAddrInfoProcess(const Record& record, CallStackString& stack);
    const std::vector<MemStateInfo>& GetPtrMemInfoList(std::pair<std::string, int64_t> key);
    std::map<std::pair<std::string, uint64_t>, std::vector<MemStateInfo>>& GetPtrMemInfoMap();
    void DeleteMemStateInfo(std::pair<std::string, uint64_t> key);
    explicit MemoryStateRecord(Config config);
private:
    void HostMemProcess(const MemOpRecord& memRecord, uint64_t& currentSize);
    void HalMemProcess(MemOpRecord& memRecord, uint64_t& currentSize, std::string& deviceType);
    MemRecordAttr GetMemInfoAttr(MemOpRecord& memRecord, uint64_t currentSize);
private:
    void PackDumpContainer(
        DumpContainer &container, const MemPoolRecord &memPool, const std::string memPoolType, MemRecordAttr &attr);
    std::map<std::pair<std::string, uint64_t>, std::vector<MemStateInfo>> ptrMemoryInfoMap_;
    std::unordered_map<uint64_t, uint64_t> hostMemSizeMap_;
    std::unordered_map<uint64_t, uint64_t> memSizeMap_;
    std::map<RecordType, HandlerFunc> memInfoProcessFuncMap_ = {
        {RecordType::MEMORY_RECORD,
            std::bind(&MemoryStateRecord::MemoryInfoProcess, this, std::placeholders::_1, std::placeholders::_2)},
        {RecordType::ATB_MEMORY_POOL_RECORD,
            std::bind(&MemoryStateRecord::MemoryPoolInfoProcess, this, std::placeholders::_1, std::placeholders::_2)},
        {RecordType::TORCH_NPU_RECORD,
            std::bind(&MemoryStateRecord::MemoryPoolInfoProcess, this, std::placeholders::_1, std::placeholders::_2)},
        {RecordType::MINDSPORE_NPU_RECORD,
            std::bind(&MemoryStateRecord::MemoryPoolInfoProcess, this, std::placeholders::_1, std::placeholders::_2)},
        {RecordType::MEM_ACCESS_RECORD,
            std::bind(&MemoryStateRecord::MemoryAccessInfoProcess, this, std::placeholders::_1, std::placeholders::_2)},
        {RecordType::ADDR_INFO_RECORD,
            std::bind(&MemoryStateRecord::MemoryAddrInfoProcess, this, std::placeholders::_1, std::placeholders::_2)},
    };
    std::mutex recordMutex_;
    Config config_;
};

}

#endif
