// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "analyzer.h"
#include "log.h"


namespace Leaks {

constexpr uint64_t MEM_MODULE_ID_BIT = 56;

// Module id
const std::unordered_map<int, std::string> g_ModuleHashTable = {
    {0, "SLOG"},          /**< Slog */
    {1, "IDEDD"},         /**< IDE daemon device */
    {2, "IDEDH"},         /**< IDE daemon host */
    {3, "HCCL"},          /**< HCCL */
    {4, "FMK"},           /**< Adapter */
    {5, "HIAIENGINE"},    /**< Matrix */
    {6, "DVPP"},          /**< DVPP */
    {7, "RUNTIME"},       /**< Runtime */
    {8, "CCE"},           /**< CCE */
    {9, "HDC"},           /**< HDC */
    {10, "DRV"},           /**< Driver */
    {11, "MDCFUSION"},     /**< Mdc fusion */
    {12, "MDCLOCATION"},   /**< Mdc location */
    {13, "MDCPERCEPTION"}, /**< Mdc perception */
    {14, "MDCFSM"},
    {15, "MDCCOMMON"},
    {16, "MDCMONITOR"},
    {17, "MDCBSWP"},    /**< MDC base software platform */
    {18, "MDCDEFAULT"}, /**< MDC undefine */
    {19, "MDCSC"},      /**< MDC spatial cognition */
    {20, "MDCPNC"},
    {21, "MLL"},      /**< abandon */
    {22, "DEVMM"},    /**< Dlog memory managent */
    {23, "KERNEL"},   /**< Kernel */
    {24, "LIBMEDIA"}, /**< Libmedia */
    {25, "CCECPU"},   /**< aicpu shedule */
    {26, "ASCENDDK"}, /**< AscendDK */
    {27, "ROS"},      /**< ROS */
    {28, "HCCP"},
    {29, "ROCE"},
    {30, "TEFUSION"},
    {31, "PROFILING"}, /**< Profiling */
    {32, "DP"},        /**< Data Preprocess */
    {33, "APP"},       /**< User Application */
    {34, "TS"},        /**< TS module */
    {35, "TSDUMP"},    /**< TSDUMP module */
    {36, "AICPU"},     /**< AICPU module */
    {37, "LP"},        /**< LP module */
    {38, "TDT"},       /**< tsdaemon or aicpu shedule */
    {39, "FE"},
    {40, "MD"},
    {41, "MB"},
    {42, "ME"},
    {43, "IMU"},
    {44, "IMP"},
    {45, "GE"}, /**< Fmk */
    {46, "MDCFUSA"},
    {47, "CAMERA"},
    {48, "ASCENDCL"},
    {49, "TEEOS"},
    {50, "ISP"},
    {51, "SIS"},
    {52, "HSM"},
    {53, "DSS"},
    {54, "PROCMGR"},     // Process Manager, Base Platform
    {55, "BBOX"},
    {56, "AIVECTOR"},
    {57, "TBE"},
    {58, "FV"},
    {59, "MDCMAP"},
    {60, "TUNE"},
    {61, "HSS"}, /**< helper */
    {62, "FFTS"},
    {63, "OP"},
    {64, "UDF"},
    {65, "HICAID"},
    {66, "TSYNC"},
    {67, "AUDIO"},
    {68, "TPRT"},
    {69, "ASCENDCKERNEL"},
    {70, "ASYS"},
    {71, "ATRACE"},
    {72, "RTC"},
    {73, "SYSMONITOR"},
    {74, "AML"},
    {75, "INVLID_MOUDLE_ID"}    // add new module before INVLID_MOUDLE_ID
};

inline int32_t GetMallocModuleId(unsigned long long flag)
{
    // bit56~63: model id
    return (flag & 0xFF00000000000000) >> MEM_MODULE_ID_BIT;
}

bool MemOpRecordKey::operator==(const MemOpRecordKey& other) const
{
    return addr_ == other.addr_;
}

std::size_t MemOpRecordKeyHash::operator()(const MemOpRecordKey& memrecordkey) const
{
    return std::hash<uint64_t>()(memrecordkey.addr_);
}

void MemoryHashTable::RecordMalloc(const ClientId &clientId, const MemOpRecord memrecord, const EventRecord &record)
{
    MemOpRecordKey memkey(memrecord.addr);
    // malloc操作需解析当前moduleId
    auto flag = record.flag;
    int32_t flagId = GetMallocModuleId(flag);
    bool foundModule = false;
    std::string modulename = "INVLID_MOUDLE_ID";
    if (g_ModuleHashTable.find(flagId) != g_ModuleHashTable.end()) {
        modulename = g_ModuleHashTable.find(flagId)->second;
        foundModule = true;
    }
    if (!foundModule) {
        Utility::LogError("[rank %u]: Malloc operator did not find %d Module in index %u malloc record.",
            clientId, flagId, memrecord.recordIndex);
    }

    Utility::LogInfo("[rank %u]: server malloc record, index: %u, addr: 0x%lx, size: %u, space: %u, module: %s",
        clientId, memrecord.recordIndex, memrecord.addr, memrecord.memSize, memrecord.space, modulename.c_str());

    if (table.find(memkey) != table.end() && (table.find(memkey)->second == 1)) {
        Utility::LogError("[rank %u]: server already has malloc record in addr: 0x%lx ,",
            " but now malloc again in index: %u, addr: 0x%lx, size: %u, space: %u",
            clientId, memrecord.addr,  memrecord.recordIndex, memrecord.addr, memrecord.memSize, memrecord.space);
    }
    table[memkey] = 1;
}

void MemoryHashTable::RecordFree(const ClientId &clientId, const MemOpRecord memrecord)
{
    MemOpRecordKey memkey(memrecord.addr);
    Utility::LogInfo("[rank %u]: server free record, index: %u, addr: 0x%lx",
        clientId, memrecord.recordIndex, memrecord.addr);

    auto it = table.find(memkey);
    if (it != table.end()) {
        if (it->second == 1) {
            table[memkey] = 0;
        } else {
            Utility::LogError("[rank %u]: Double free operator found for malloc operation : addr: 0x%lx",
                clientId, memrecord.addr);
        }
    } else {
            Utility::LogError("[rank %u]: No matching malloc operation found for free operator: addr: 0x%lx",
                clientId, memrecord.addr);
    }
}

void MemoryHashTable::Record(const ClientId &clientId, const EventRecord &record)
{
    auto memrecord = record.record.memoryRecord;
    if (memrecord.memType == MemOpType::MALLOC) {
        RecordMalloc(clientId, memrecord, record);
    } else if (memrecord.memType == MemOpType::FREE) {
        RecordFree(clientId, memrecord);
    }
    return;
}

void MemoryHashTable::CheckLeak(const size_t clientId)
{
    bool foundLeaks = false;
    for (const auto& pair :table) {
        if (pair.second != 0) {
            foundLeaks = true;
            Utility::LogWarn("[rank %u]: Leak memory in Malloc operator, addr: 0x%lx", clientId, pair.first.addr_);
        }
    }
    if (!foundLeaks) {
        Utility::LogInfo("[rank %u]: There is no leak memory.", clientId);
    }
}

Analyzer::Analyzer(const AnalysisConfig &config)
{
    config_ = config;
}

MemoryHashTable& Analyzer::GetMemTable(const ClientId &clientId)
{
    int32_t tableIndex = clientId;
    if (tableIndex >= memtablelist.size()) {
        memtablelist.resize(tableIndex + 1);
    }
    return memtablelist[tableIndex];
}

void Analyzer::Do(const ClientId &clientId, const EventRecord &record)
{
    switch (record.type) {
        case RecordType::MEMORY_RECORD: {
            auto& memhashtable = GetMemTable(clientId);
            memhashtable.Record(clientId, record);
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
    if (memtablelist.empty()) {
        Utility::LogError("No memory records available.");
    }
    for (int32_t tableIndex = 0; tableIndex< memtablelist.size(); ++tableIndex) {
        memtablelist[tableIndex].CheckLeak(tableIndex);
    }

    return;
}

Analyzer::~Analyzer()
{
    LeakAnalyze();
}

}
