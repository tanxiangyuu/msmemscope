// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "trace_record.h"
#include "log.h"
#include "framework/record_info.h"
#include "utils.h"
#include "file.h"
#include "umask_guard.h"
#include "config_info.h"

namespace Leaks {
constexpr uint32_t DEFAULT_UMASK_FOR_JSON_FILE = 0177;

inline std::string FormatCompleteEvent(JsonBaseInfo baseInfo, uint64_t dur, std::string args = "")
{
    std::ostringstream oss;
    oss << "{\n"
        << "    \"ph\": \"X\",\n"
        << "    \"name\": \"" << baseInfo.name << "\",\n"
        << "    \"pid\": " << baseInfo.pid << ",\n"
        << "    \"tid\": " << baseInfo.tid << ",\n"
        << "    \"ts\": " << baseInfo.ts << ",\n"
        << "    \"dur\": " << dur;
    if (!args.empty()) {
        oss << ",\n"
            << "    \"args\": {\n"
            << "        " << args << "\n"
            << "    }";
    }
    oss << "\n},\n";
    return oss.str();
}

inline std::string FormatCounterEvent(JsonBaseInfo baseInfo, std::string size)
{
    std::ostringstream oss;
    oss << "{\n"
        << "    \"ph\": \"C\",\n"
        << "    \"name\": \"" << baseInfo.name << "\",\n"
        << "    \"pid\": " << baseInfo.pid << ",\n"
        << "    \"tid\": " << baseInfo.tid << ",\n"
        << "    \"ts\": " << baseInfo.ts << ",\n"
        << "    \"args\": {\n"
        << "        \"size\": " << size << "\n"
        << "    }\n"
        << "},\n";
    return oss.str();
}

inline std::string FormatInstantEvent(JsonBaseInfo baseInfo, std::string args = "")
{
    std::ostringstream oss;
    oss << "{\n"
        << "    \"ph\": \"i\",\n"
        << "    \"name\": \"" << baseInfo.name << "\",\n"
        << "    \"pid\": " << baseInfo.pid << ",\n"
        << "    \"tid\": " << baseInfo.tid << ",\n"
        << "    \"ts\": " << baseInfo.ts << ",\n"
        << "    \"s\": \"p\"";
    if (!args.empty()) {
        oss << ",\n"
            << "    \"args\": {\n"
            << "        \"message\": \"" << args << "\"\n"
            << "    }";
    }
    oss << "\n},\n";
    return oss.str();
}

inline std::string FormatMetadataEvent(JsonBaseInfo baseInfo, std::string args)
{
    std::ostringstream oss;
    oss << "{\n"
        << "    \"ph\": \"M\",\n"
        << "    \"name\": \"" << baseInfo.name << "\",\n"
        << "    \"pid\": " << baseInfo.pid << ",\n"
        << "    \"tid\": " << baseInfo.tid << ",\n"
        << "    \"args\": {\n"
        << "        " << args << "\n"
        << "    }\n"
        << "},\n";
    return oss.str();
}

TraceRecord& TraceRecord::GetInstance()
{
    static TraceRecord instance;
    return instance;
}

TraceRecord::TraceRecord()
{
    eventPids_.emplace_back(EventPid{mstxEventPid_, "mstx"});
    eventPids_.emplace_back(EventPid{leakEventPid_, "leak"});
}

void TraceRecord::TraceHandler(const EventRecord &record)
{
    if (!ProcessRecord(record)) {
        Utility::LogWarn("Record was not processed correctly.");
    }
}

bool TraceRecord::CreateFileWithDeviceId(const int32_t &devId)
{
    if (traceFiles_[devId].fp != nullptr) {
        return true;
    }

    if (!Utility::MakeDir(OUTPUT_DIR_PATH)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(createFileMutex_);

    std::string filePath = OUTPUT_DIR_PATH + "/device" + std::to_string(devId) +
            "_trace_" + Utility::GetDateStr() + ".json";
    Utility::UmaskGuard guard{DEFAULT_UMASK_FOR_JSON_FILE};
    FILE* fp = fopen(filePath.c_str(), "a");
    if (fp != nullptr) {
        fprintf(fp, "[\n");
        traceFiles_[devId].fp = fp;
        traceFiles_[devId].filePath = filePath;
    } else {
        Utility::LogError("Device %d open file %s error", devId, filePath.c_str());
        return false;
    }

    return true;
}

bool TraceRecord::CheckStrHasContent(const std::string &str)
{
    if (str == "") {
        return false;
    }
    return true;
}

void TraceRecord::SafeWriteString(const std::string &str, const int32_t &devId)
{
    if (devId < 0) {
        return;
    }
    if (!CreateFileWithDeviceId(devId)) {
        Utility::LogError("Create file for device %d failed.", devId);
        return;
    }
    std::lock_guard<std::mutex> lock(writeFileMutex_[devId]);
    fprintf(traceFiles_[devId].fp, "%s", str.c_str());
}

void TraceRecord::ProcessTorchMemLeakInfo(const TorchMemLeakInfo &info)
{
    std::string str;
    TorchMemLeakInfoToString(info, str);
    SafeWriteString(str, info.devId);
    return;
}

void TraceRecord::TorchMemLeakInfoToString(const TorchMemLeakInfo &info, std::string &str)
{
    uint64_t tid = 0;
    std::string args = "\"addr\": " + std::to_string(info.addr) + ",\"size\": " + std::to_string(info.size);
    JsonBaseInfo baseInfo{"mem " + std::to_string(info.addr) + " leak", leakEventPid_, tid, info.timestamp};
    str = FormatCompleteEvent(baseInfo, info.duration, args);
}

bool TraceRecord::ProcessRecord(const EventRecord &record)
{
    std::string str = "";
    int32_t devId = -1;
    switch (record.type) {
        case RecordType::MEMORY_RECORD: {
            auto memRecord = record.record.memoryRecord;
            devId = memRecord.devId;
            RecordToString(memRecord, str);
            break;
        }
        case RecordType::KERNEL_LAUNCH_RECORD: {
            auto kernelLaunchRecord = record.record.kernelLaunchRecord;
            devId = kernelLaunchRecord.devId;
            RecordToString(kernelLaunchRecord, str);
            break;
        }
        case RecordType::ACL_ITF_RECORD: {
            auto aclItfRecord = record.record.aclItfRecord;
            devId = aclItfRecord.devId;
            RecordToString(aclItfRecord, str);
            break;
        }
        case RecordType::TORCH_NPU_RECORD: {
            TorchNpuRecord torchNpuRecord = record.record.torchNpuRecord;
            devId = torchNpuRecord.devId;
            RecordToString(torchNpuRecord, str);
            break;
        }
        case RecordType::MSTX_MARK_RECORD: {
            MstxRecord mstxRecord = record.record.mstxRecord;
            devId = mstxRecord.devId;
            RecordToString(mstxRecord, str);
            break;
        }
        default:
            break;
    }

    if (!CheckStrHasContent(str)) {
        return false;
    }
    SafeWriteString(str, devId);
    return true;
}

void TraceRecord::RecordToString(const MemOpRecord &memRecord, std::string &str)
{
    int32_t devId = memRecord.devId;
    uint64_t addr = memRecord.addr;
    uint64_t size = memRecord.memSize;

    // 最终统计的结果只包含device的数据
    if (memRecord.space == MemOpSpace::DEVICE && memRecord.memType == MemOpType::MALLOC) {
        deviceMemAllocation_[devId][addr] = size;
        deviceMemUsage_[devId] = Utility::GetAddResult(deviceMemUsage_[devId], size);
    } else if (memRecord.space == MemOpSpace::INVALID) {
        if (deviceMemAllocation_[devId].find(addr) == deviceMemAllocation_[devId].end()) {
            Utility::LogWarn("No memory allocation record for the freed addr %llx.", addr);
            return;
        }
        deviceMemUsage_[devId] = Utility::GetSubResult(deviceMemUsage_[devId], deviceMemAllocation_[devId][addr]);
        deviceMemAllocation_[devId].erase(addr);
    } else {
        return;
    }

    truePids_[devId].insert(memRecord.pid);
    JsonBaseInfo baseInfo{"device memory", memRecord.pid, memRecord.tid, memRecord.timeStamp};
    str = FormatCounterEvent(baseInfo, std::to_string(deviceMemUsage_[devId]));
    return;
}

void TraceRecord::RecordToString(const KernelLaunchRecord &kernelLaunchRecord, std::string &str)
{
    JsonBaseInfo baseInfo{
        "kernel_" + std::to_string(kernelLaunchRecord.kernelLaunchIndex),
        kernelLaunchRecord.pid,
        kernelLaunchRecord.tid,
        kernelLaunchRecord.timeStamp
    };
    str = FormatInstantEvent(baseInfo);
    return;
}

void TraceRecord::RecordToString(const AclItfRecord &aclItfRecord, std::string &str)
{
    if (aclItfRecord.devId == GD_INVALID_NUM) {
        return;
    }
    std::string name = aclItfRecord.type == AclOpType::INIT ? "acl_init" : "acl_finalize";
    JsonBaseInfo baseInfo{
        name,
        aclItfRecord.tid,
        aclItfRecord.tid,
        aclItfRecord.timeStamp
    };
    str = FormatInstantEvent(baseInfo);
    return;
}

void TraceRecord::RecordToString(const TorchNpuRecord &torchNpuRecord, std::string &str)
{
    MemoryUsage memoryUsage = torchNpuRecord.memoryUsage;
    uint64_t timestamp = torchNpuRecord.timeStamp;
    uint64_t pid = torchNpuRecord.pid;
    uint64_t tid = torchNpuRecord.tid;

    truePids_[torchNpuRecord.devId].insert(pid);
    JsonBaseInfo reservedBaseInfo{"operators reserved", pid, tid, timestamp};
    JsonBaseInfo activeBaseInfo{"operators active", pid, tid, timestamp};
    JsonBaseInfo allocatedBaseInfo{"operators allocated", pid, tid, timestamp};
    
    str = FormatCounterEvent(reservedBaseInfo, std::to_string(memoryUsage.totalReserved));
    str += FormatCounterEvent(activeBaseInfo, std::to_string(memoryUsage.totalActive));
    str += FormatCounterEvent(allocatedBaseInfo, std::to_string(memoryUsage.totalAllocated));
    
    return;
}

void TraceRecord::RecordToString(const MstxRecord &mstxRecord, std::string &str)
{
    int32_t devId = mstxRecord.devId;
    std::string mstxEventName;
    if (mstxRecord.markType == MarkType::MARK_A) {
        mstxEventName = "mstx_mark";
    } else if (mstxRecord.markType == MarkType::RANGE_START_A) {
        if (strcmp(mstxRecord.markMessage, "step start") == 0) {
            mstxEventName = "mstx_step" + std::to_string(mstxRecord.stepId) + "_start";
            stepStartTime_[devId][mstxRecord.rangeId] = mstxRecord.timeStamp;
        } else {
            mstxEventName = "mstx_range" + std::to_string(mstxRecord.rangeId) + "_start";
        }
    } else {
        if (stepStartTime_.find(devId) == stepStartTime_.end() ||
            stepStartTime_[devId].find(mstxRecord.rangeId) == stepStartTime_[devId].end()) {
            mstxEventName = "mstx_range" + std::to_string(mstxRecord.rangeId) + "_end";
        } else {
            mstxEventName = "mstx_step" + std::to_string(mstxRecord.stepId) + "_end";
            JsonBaseInfo stepBaseInfo{
                "step " + std::to_string(mstxRecord.stepId),
                mstxEventPid_,
                mstxRecord.tid,
                stepStartTime_[devId][mstxRecord.rangeId]
            };
            str = FormatCompleteEvent(stepBaseInfo, mstxRecord.timeStamp - stepStartTime_[devId][mstxRecord.rangeId]);
        }
    }

    JsonBaseInfo baseInfo{
        mstxEventName,
        mstxEventPid_,
        mstxRecord.tid,
        mstxRecord.timeStamp
    };
    str += FormatInstantEvent(baseInfo, mstxRecord.markMessage);
    return;
}

void TraceRecord::SetMetadataEvent(const int32_t &devId)
{
    std::string str;
    uint64_t sortIndex = 0;

    for (auto pid : truePids_[devId]) {
        JsonBaseInfo sortBaseInfo{"process_sort_index", pid, 0, 0};
        str += FormatMetadataEvent(sortBaseInfo, "\"sort_index\": " + std::to_string(sortIndex++));
    }

    for (auto eventPid : eventPids_) {
        JsonBaseInfo nameBaseInfo{"process_name", eventPid.pid, 0, 0};
        str += FormatMetadataEvent(nameBaseInfo, "\"name\": \"" + eventPid.name+ "\"");

        JsonBaseInfo sortBaseInfo{"process_sort_index", eventPid.pid, 0, 0};
        str += FormatMetadataEvent(sortBaseInfo, "\"sort_index\": " + std::to_string(sortIndex++));
    }

    SafeWriteString(str, devId);
}

// TraceRecord生命周期结束时，文件写入完毕，关闭文件
TraceRecord::~TraceRecord()
{
    for (auto &file : traceFiles_) {
        FILE *fp = file.second.fp;
        if (fp != nullptr) {
            SetMetadataEvent(file.first);
            fprintf(fp, "{\n}\n]");
            fclose(fp);
        }
    }
}
}