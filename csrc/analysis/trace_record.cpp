// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "trace_record.h"
#include "log.h"
#include "framework/record_info.h"
#include "utils.h"
#include "file.h"

namespace Leaks {

constexpr uint64_t TRACE_DURATION = 10;   // 为便于可视化aclItf、kernel launch这两类瞬时事件，生成json文件时将其设定有10微秒的持续时间

static std::string FormatCompleteEvent(JsonBaseInfo baseInfo, uint64_t dur)
{
    std::ostringstream oss;
    oss << "{\n"
        << "    \"ph\": \"X\",\n"
        << "    \"name\": \"" << baseInfo.name << "\",\n"
        << "    \"pid\": " << baseInfo.pid << ",\n"
        << "    \"tid\": " << baseInfo.tid << ",\n"
        << "    \"ts\": " << baseInfo.ts << ",\n"
        << "    \"dur\": " << dur << "\n"
        << "},\n";
    return oss.str();
}

static std::string FormatCompleteEventWithArgs(JsonBaseInfo baseInfo, uint64_t dur, std::string args)
{
    std::ostringstream oss;
    oss << "{\n"
        << "    \"ph\": \"X\",\n"
        << "    \"name\": \"" << baseInfo.name << "\",\n"
        << "    \"pid\": " << baseInfo.pid << ",\n"
        << "    \"tid\": " << baseInfo.tid << ",\n"
        << "    \"ts\": " << baseInfo.ts << ",\n"
        << "    \"dur\": " << dur << ",\n"
        << "    \"args\": {\n"
        << "        " << args << "\n"
        << "    }\n"
        << "},\n";
    return oss.str();
}

static std::string FormatCounterEvent(JsonBaseInfo baseInfo, uint64_t size)
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

static std::string FormatInstantEvent(JsonBaseInfo baseInfo)
{
    std::ostringstream oss;
    oss << "{\n"
        << "    \"ph\": \"i\",\n"
        << "    \"name\": \"" << baseInfo.name << "\",\n"
        << "    \"pid\": " << baseInfo.pid << ",\n"
        << "    \"tid\": " << baseInfo.tid << ",\n"
        << "    \"ts\": " << baseInfo.ts << ",\n"
        << "    \"s\": \"p\"\n"
        << "},\n";
    return oss.str();
}

TraceRecord& TraceRecord::GetInstance()
{
    static TraceRecord instance;
    return instance;
}

void TraceRecord::TraceHandler(const EventRecord &record)
{
    if (!ProcessRecord(record)) {
        Utility::LogWarn("Write record in trace file failed.");
    }
}

bool TraceRecord::CreateFileWithDeviceId(const int32_t &devId)
{
    if (traceFiles_[devId].fp != nullptr) {
        return true;
    }

    std::string dirPath = "leaksDumpResults";
    if (!Utility::MakeDir(dirPath)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(createFileMutex_);

    std::string filePath = dirPath + "/device" + std::to_string(devId) + "_trace_" + Utility::GetDateStr() + ".json";
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
    std::lock_guard<std::mutex> lock(writeFileMutex_[devId]);
    fprintf(traceFiles_[devId].fp, "%s", str.c_str());
}

void TraceRecord::ProcessTorchMemLeakInfo(const TorchMemLeakInfo &info)
{
    std::string str;
    TorchMemLeakInfoToString(info, str);

    int32_t devId = info.devId;
    if (devId < 0) {
        return;
    }
    if (!CreateFileWithDeviceId(devId)) {
        Utility::LogError("Create file for device %d failed.", devId);
        return;
    }
    SafeWriteString(str, devId);
    return;
}

void TraceRecord::TorchMemLeakInfoToString(const TorchMemLeakInfo &info, std::string &str)
{
    std::string args = "\"addr\": " + std::to_string(info.addr) + ",\"size\": " + std::to_string(info.size);
    JsonBaseInfo baseInfo{"mem " + std::to_string(info.addr) + " leak", 0, 0, info.timestamp};
    str = FormatCompleteEventWithArgs(baseInfo, info.duration, args);
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

    if (!CheckStrHasContent(str) || (devId < 0)) {
        return false;
    }
    if (!CreateFileWithDeviceId(devId)) {
        Utility::LogError("Create file for device %d failed.", devId);
        return false;
    }
    SafeWriteString(str, devId);
    return true;
}

void TraceRecord::RecordToString(const MemOpRecord &memRecord, std::string &str)
{
    int32_t devId = memRecord.devId;
    std::string space;
    uint64_t totalMem;

    if (memRecord.space == MemOpSpace::HOST) {
        if (memRecord.memType == MemOpType::MALLOC) {
            hostMemAllocation_[devId][memRecord.addr] = memRecord.memSize;
            if (hostMemUsage_[devId] > UINT64_MAX - memRecord.memSize) {
                Utility::LogError("Host hal memory overflow.");
                return;
            }
            hostMemUsage_[devId] += memRecord.memSize;
        } else if (hostMemAllocation_[devId].find(memRecord.addr) != hostMemAllocation_[devId].end()) {
            hostMemUsage_[devId] -= hostMemAllocation_[devId][memRecord.addr];
            hostMemAllocation_[devId].erase(memRecord.addr);
        } else {
            Utility::LogWarn("No memory allocation record for the freed addr %llu on host.", memRecord.addr);
            return;
        }
        space = "host memory";
        totalMem = hostMemUsage_[devId];
    } else if (memRecord.space == MemOpSpace::DEVICE) {
        if (memRecord.memType == MemOpType::MALLOC) {
            deviceMemAllocation_[devId][memRecord.addr] = memRecord.memSize;
            if (deviceMemUsage_[devId] > UINT64_MAX - memRecord.memSize) {
                Utility::LogError("Device hal memory overflow.");
                return;
            }
            deviceMemUsage_[devId] += memRecord.memSize;
        } else if (deviceMemAllocation_[devId].find(memRecord.addr) != deviceMemAllocation_[devId].end()) {
            deviceMemUsage_[devId] -= deviceMemAllocation_[devId][memRecord.addr];
            deviceMemAllocation_[devId].erase(memRecord.addr);
        } else {
            Utility::LogWarn("No memory allocation record for the freed addr %llu on device.", memRecord.addr);
            return;
        }
        space = "device memory";
        totalMem = deviceMemUsage_[devId];
    } else {
        Utility::LogWarn("Not sure whether the hal memory record is on the host or device.");
        return;
    }

    JsonBaseInfo baseInfo{space.c_str(), memRecord.pid, memRecord.tid, memRecord.timeStamp};
    str = FormatCounterEvent(baseInfo, totalMem);
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
    str = FormatCompleteEvent(baseInfo, TRACE_DURATION);
    return;
}

void TraceRecord::RecordToString(const AclItfRecord &aclItfRecord, std::string &str)
{
    JsonBaseInfo baseInfo{
        "acl_" + std::to_string(aclItfRecord.aclItfRecordIndex),
        aclItfRecord.pid,
        aclItfRecord.tid,
        aclItfRecord.timeStamp
    };
    str = FormatCompleteEvent(baseInfo, TRACE_DURATION);
    return;
}

void TraceRecord::RecordToString(const TorchNpuRecord &torchNpuRecord, std::string &str)
{
    MemoryUsage memoryUsage = torchNpuRecord.memoryUsage;
    uint64_t timestamp = torchNpuRecord.timeStamp;
    uint64_t pid = torchNpuRecord.pid;
    uint64_t tid = torchNpuRecord.tid;

    JsonBaseInfo reservedBaseInfo{"operators reserved", pid, tid, timestamp};
    JsonBaseInfo activeBaseInfo{"operators active", pid, tid, timestamp};
    JsonBaseInfo allocatedBaseInfo{"operators allocated", pid, tid, timestamp};
    
    str = FormatCounterEvent(reservedBaseInfo, static_cast<uint64_t>(memoryUsage.totalReserved));
    str += FormatCounterEvent(activeBaseInfo, static_cast<uint64_t>(memoryUsage.totalActive));
    str += FormatCounterEvent(allocatedBaseInfo, static_cast<uint64_t>(memoryUsage.totalAllocated));
    
    return;
}

void TraceRecord::RecordToString(const MstxRecord &mstxRecord, std::string &str)
{
    int32_t devId = mstxRecord.devId;
    std::string mstxType;
    if (mstxRecord.markType == MarkType::MARK_A) {
        mstxType = "mstx_mark";
    } else {
        if (mstxRecord.markType == MarkType::RANGE_START_A) {
            mstxType = "mstx_range_start";
            lastStepStartTime_[devId] = mstxRecord.timeStamp;
        } else if (mstxRecord.markType == MarkType::RANGE_END) {
            mstxType = "mstx_range_end";
            JsonBaseInfo rangeBaseInfo{
                "step " + std::to_string(mstxRecord.rangeId),
                mstxRecord.pid,
                mstxRecord.tid,
                lastStepStartTime_[devId]
            };
            str = FormatCompleteEvent(rangeBaseInfo, mstxRecord.timeStamp - lastStepStartTime_[devId]);
        }
    }

    JsonBaseInfo baseInfo{
        mstxType + "_" + std::to_string(mstxRecord.rangeId),
        mstxRecord.pid,
        mstxRecord.tid,
        mstxRecord.timeStamp
    };
    str += FormatInstantEvent(baseInfo);
    return;
}

// TraceRecord生命周期结束时，文件写入完毕，关闭文件
TraceRecord::~TraceRecord()
{
    for (auto &file : traceFiles_) {
        FILE *fp = file.second.fp;
        if (fp != nullptr) {
            fprintf(fp, "{\n}\n]");
            fclose(fp);
        }
    }
}
}