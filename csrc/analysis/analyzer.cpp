// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "analyzer.h"
#include "log.h"

namespace Leaks {



constexpr uint64_t MEM_MODULE_ID_BIT = 56;
constexpr uint64_t MEM_MODULE_ID_WIDTH = 8;
constexpr uint64_t MEM_MODULE_ID_MASK = ((1UL << MEM_MODULE_ID_WIDTH) - 1) << MEM_MODULE_ID_BIT;

//Module id
const char* ModuleNames[] = {
    "SLOG",          /**< Slog */
    "IDEDD",         /**< IDE daemon device */
    "IDEDH",         /**< IDE daemon host */
    "HCCL",          /**< HCCL */
    "FMK",           /**< Adapter */
    "HIAIENGINE",    /**< Matrix */
    "DVPP",          /**< DVPP */
    "RUNTIME",       /**< Runtime */
    "CCE",           /**< CCE */
    "HDC",           /**< HDC */
    "DRV",           /**< Driver */
    "MDCFUSION",     /**< Mdc fusion */
    "MDCLOCATION",   /**< Mdc location */
    "MDCPERCEPTION", /**< Mdc perception */
    "MDCFSM",
    "MDCCOMMON",
    "MDCMONITOR",
    "MDCBSWP",    /**< MDC base software platform */
    "MDCDEFAULT", /**< MDC undefine */
    "MDCSC",      /**< MDC spatial cognition */
    "MDCPNC",
    "MLL",      /**< abandon */
    "DEVMM",    /**< Dlog memory managent */
    "KERNEL",   /**< Kernel */
    "LIBMEDIA", /**< Libmedia */
    "CCECPU",   /**< aicpu shedule */
    "ASCENDDK", /**< AscendDK */
    "ROS",      /**< ROS */
    "HCCP",
    "ROCE",
    "TEFUSION",
    "PROFILING", /**< Profiling */
    "DP",        /**< Data Preprocess */
    "APP",       /**< User Application */
    "TS",        /**< TS module */
    "TSDUMP",    /**< TSDUMP module */
    "AICPU",     /**< AICPU module */
    "LP",        /**< LP module */
    "TDT",       /**< tsdaemon or aicpu shedule */
    "FE",
    "MD",
    "MB",
    "ME",
    "IMU",
    "IMP",
    "GE", /**< Fmk */
    "MDCFUSA",
    "CAMERA",
    "ASCENDCL",
    "TEEOS",
    "ISP",
    "SIS",
    "HSM",
    "DSS",
    "PROCMGR",     // Process Manager, Base Platform
    "BBOX",
    "AIVECTOR",
    "TBE",
    "FV",
    "MDCMAP",
    "TUNE",
    "HSS", /**< helper */
    "FFTS",
    "OP",
    "UDF",
    "HICAID",
    "TSYNC",
    "AUDIO",
    "TPRT",
    "ASCENDCKERNEL",
    "ASYS",
    "ATRACE",
    "RTC",
    "SYSMONITOR",
    "AML",
    "INVLID_MOUDLE_ID"    // add new module before INVLID_MOUDLE_ID
};

inline int32_t GetMallocModuleId(unsigned long long flag)
{   
    return (flag & 0xFF00000000000000) >> MEM_MODULE_ID_BIT;
    //return flag & MEM_MODULE_ID_MASK;
}


MemOpRecordKey::MemOpRecordKey(const uint64_t &addr)
{
    addr_ = addr;

} 

bool MemOpRecordKey::operator==(const MemOpRecordKey& other) const {
    return addr_ == other.addr_;
}

std::size_t MemOpRecordKeyHash::operator()(const MemOpRecordKey& memrecordkey) const {
        return std::hash<uint64_t>()(memrecordkey.addr_);
}


void MemoryHashTable::Record(const EventRecord &record)
{
    auto memrecord = record.record.memoryRecord;
    MemOpRecordKey memkey(memrecord.addr);
    if (memrecord.memType == MemOpType::MALLOC) {

        // malloc操作需解析当前moduleId
        auto flag = record.flag;
        //Utility::LogInfo("flag: %llu", flag);
        int32_t flagId = GetMallocModuleId(flag);
        bool found_module = false;
        for (int i = static_cast<int>(ModuleID::SLOG); i<= static_cast<int>(ModuleID::INVLID_MOUDLE_ID); ++i) {
            if(flagId == i){
                found_module = true;
                Utility::LogInfo("Malloc operator in %s Module.", ModuleNames[i]);
                break;
            }
        }
        if(!found_module){
            Utility::LogError("Malloc operator did not find %d Module.", flagId);
        }

        Utility::LogInfo("server malloc record, index: %u, addr: 0x%lx, size: %u, space: %u",
        memrecord.recordIndex, memrecord.addr, memrecord.memSize, memrecord.space);

        //检测是否该地址是否已有申请
        if (table.find(memkey) != table.end()) {
            Utility::LogError("server already has malloc record in addr: 0x%lx , but now malloc again in index: ",
            "%u, addr: 0x%lx, size: %u, space: %u",memrecord.addr,  memrecord.recordIndex, memrecord.addr, 
            memrecord.memSize, memrecord.space);
        }
        // malloc，插入记录
        table[memkey] = memrecord.memType;
    } else if (memrecord.memType == MemOpType::FREE){
        Utility::LogInfo("server free record, index: %u, addr: 0x%lx", 
        memrecord.recordIndex, memrecord.addr);
        // free, 删除记录
        auto it = table.find(memkey);
        if (it != table.end() && it->second == MemOpType::MALLOC){
            table.erase(it);
        } else {
             Utility::LogError("No matching malloc operation found for free operator: addr: 0x%lx",
             memrecord.addr);
        }
    } else {
         Utility::LogError("Invalid memType.");
    }

    return;
}

void MemoryHashTable::CheckLeak()
{
    if (table.empty()) {
        Utility::LogInfo("There is no leak memory.");
    } else {
        for (const auto& pair :table) {
            Utility::LogWarn("Leak memory in Malloc operator, addr: 0x%lx", pair.first.addr_);
        }
    }
}

Analyzer::Analyzer(const AnalysisConfig &config)
{
    config_ = config;
}

void Analyzer::Do(const EventRecord &record)
{   
    memhashtable.Record(record);
    return;
}

void Analyzer::LeakAnalyze()
{
    memhashtable.CheckLeak();
}


}