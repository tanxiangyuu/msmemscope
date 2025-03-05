// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "trace_record.h"
#include "log.h"
#include "framework/record_info.h"
#include "utils.h"
#include "file.h"
#include "umask_guard.h"

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

inline std::string FormatDeviceName(Device device)
{
    return device.type == DeviceType::CPU ? "cpu" : "npu" + std::to_string(device.index);
}

TraceRecord& TraceRecord::GetInstance()
{
    static TraceRecord instance;
    return instance;
}

TraceRecord::TraceRecord()
{
    SetDirPath();
    eventPids_.emplace_back(EventPid{mstxEventPid_, "mstx"});
    eventPids_.emplace_back(EventPid{leakEventPid_, "leak"});
}

void TraceRecord::TraceHandler(const EventRecord &record)
{
    ProcessRecord(record);
}

void TraceRecord::SetDirPath()
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    dirPath_ = Utility::g_dirPath + "/" + std::string(TRACE_FILE);
}

bool TraceRecord::CreateFileByDevice(const Device &device)
{
    if (traceFiles_[device].fp != nullptr) {
        return true;
    }

    if (!Utility::MakeDir(dirPath_)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(fileMutex_);

    std::string fileHead = FormatDeviceName(device);
    std::string filePath = dirPath_ + "/" + fileHead + "_trace_" + Utility::GetDateStr() + ".json";

    // 校验文件路径是否合法
    if (!Utility::CheckIsValidPath(dirPath_)) {
        Utility::LogError("Device %s invalid file path %s", fileHead.c_str(), dirPath_.c_str());
        return false;
    }

    Utility::UmaskGuard guard{DEFAULT_UMASK_FOR_JSON_FILE};
    FILE* fp = fopen(filePath.c_str(), "a");
    if (fp != nullptr) {
        std::cout << "[msleaks] Info: create file " << filePath.c_str() << "." << std::endl;
        fprintf(fp, "[\n");
        traceFiles_[device].fp = fp;
        traceFiles_[device].filePath = filePath;
    } else {
        Utility::LogError("Device %s open file %s error", fileHead.c_str(), filePath.c_str());
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

void TraceRecord::SafeWriteString(const std::string &str, const Device &device)
{
    if (device.index == GD_INVALID_NUM || device.index < 0) {
        Utility::LogWarn("Invalid device id %d.", device.index);
        return;
    }
    if (!CreateFileByDevice(device)) {
        Utility::LogError("Create file for device %s failed.", FormatDeviceName(device).c_str());
        return;
    }

    std::lock_guard<std::mutex> lock(writeFileMutex_[device]);
    if (traceFiles_[device].fp != nullptr) {
        fprintf(traceFiles_[device].fp, "%s", str.c_str());
    }
}

void TraceRecord::SaveKernelLaunchRecordToCpuTrace(const std::string &str)
{
    SafeWriteString(str, {DeviceType::CPU, 0});
}

void TraceRecord::ProcessTorchMemLeakInfo(const TorchMemLeakInfo &info)
{
    std::string str;
    TorchMemLeakInfoToString(info, str);
    SafeWriteString(str, Device{DeviceType::NPU, info.devId});
    return;
}

void TraceRecord::TorchMemLeakInfoToString(const TorchMemLeakInfo &info, std::string &str)
{
    uint64_t tid = 0;
    std::string args = "\"addr\": " + std::to_string(info.addr) + ",\"size\": " + std::to_string(info.size);
    JsonBaseInfo baseInfo{"mem " + std::to_string(info.addr) + " leak", leakEventPid_, tid, info.kernelIndex};
    str = FormatCompleteEvent(baseInfo, info.duration, args);
}

void TraceRecord::ProcessRecord(const EventRecord &record)
{
    std::string str = "";
    Device device{DeviceType::NPU, GD_INVALID_NUM};

    switch (record.type) {
        case RecordType::MEMORY_RECORD: {
            auto memRecord = record.record.memoryRecord;
            MemRecordToString(memRecord, str);
            device.type = memRecord.devType;
            device.index = memRecord.devId;
            break;
        }
        case RecordType::KERNEL_LAUNCH_RECORD: {
            auto kernelLaunchRecord = record.record.kernelLaunchRecord;
            device.index = kernelLaunchRecord.devId;
            KernelLaunchRecordToString(kernelLaunchRecord, str);
            // kernellaunch record should be shown in cpu_trace and npu_trace simutanously
            SaveKernelLaunchRecordToCpuTrace(str);
            break;
        }
        case RecordType::ACL_ITF_RECORD: {
            auto aclItfRecord = record.record.aclItfRecord;
            device.index = aclItfRecord.devId;
            AclItfRecordToString(aclItfRecord, str);
            break;
        }
        case RecordType::TORCH_NPU_RECORD: {
            TorchNpuRecord torchNpuRecord = record.record.torchNpuRecord;
            device.index = torchNpuRecord.devId;
            TorchRecordToString(torchNpuRecord, str);
            break;
        }
        case RecordType::MSTX_MARK_RECORD: {
            MstxRecord mstxRecord = record.record.mstxRecord;
            device.index = mstxRecord.devId;
            MstxRecordToString(mstxRecord, str);
            break;
        }
        default:
            break;
    }

    if (!CheckStrHasContent(str)) {
        return;
    }
    SafeWriteString(str, device);
    return;
}

void TraceRecord::NpuMemRecordToString(MemOpRecord &memRecord, std::string &str)
{
    int32_t devId = memRecord.devId;
    uint64_t addr = memRecord.addr;
    uint64_t size = memRecord.memSize;
    MemOpSpace space = memRecord.space;
    bool isHost = false;

    std::lock_guard<std::mutex> lock(halMemMutex_);
    if (memRecord.memType == MemOpType::MALLOC) {
        if (space == MemOpSpace::DEVICE) {
            halDeviceMemAllocation_[addr] = MemAllocationInfo{size, devId};
            halDeviceMemUsage_[devId] = Utility::GetAddResult(halDeviceMemUsage_[devId], size);
        } else if (space == MemOpSpace::HOST) {
            halHostMemAllocation_[addr] = MemAllocationInfo{size, devId};
            halHostMemUsage_[devId] = Utility::GetAddResult(halHostMemUsage_[devId], size);
            isHost = true;
        } else {
            Utility::LogWarn("Invalid space.");
            return;
        }
    } else {
        if (halDeviceMemAllocation_.find(addr) != halDeviceMemAllocation_.end()) {
            devId = halDeviceMemAllocation_[addr].devId;
            halDeviceMemUsage_[devId] = Utility::GetSubResult(halDeviceMemUsage_[devId],
                                                              halDeviceMemAllocation_[addr].size);
            halDeviceMemAllocation_.erase(addr);
        } else if (halHostMemAllocation_.find(addr) != halHostMemAllocation_.end()) {
            devId = halHostMemAllocation_[addr].devId;
            halHostMemUsage_[devId] = Utility::GetSubResult(halHostMemUsage_[devId],
                                                            halHostMemAllocation_[addr].size);
            halHostMemAllocation_.erase(addr);
            isHost = true;
        } else {
            Utility::LogWarn("Invalid free addr %llx.", addr);
            return;
        }
    }

    std::string spaceName = isHost ? "pin memory" : "device memory";
    uint64_t memUsage = isHost ? halHostMemUsage_[devId] : halDeviceMemUsage_[devId];
    JsonBaseInfo baseInfo{spaceName, memRecord.pid, memRecord.tid, memRecord.kernelIndex};
    str = FormatCounterEvent(baseInfo, std::to_string(memUsage));

    if (isHost) {
        memRecord.devType = DeviceType::CPU;
        memRecord.devId = 0;
    } else {
        memRecord.devId = devId;
    }
    return;
}

void TraceRecord::CpuMemRecordToString(const MemOpRecord &memRecord, std::string &str)
{
    uint64_t addr = memRecord.addr;
    uint64_t size = memRecord.memSize;
    MemOpSpace space = memRecord.space;

    std::lock_guard<std::mutex> lock(hostMemMutex_);
    if (memRecord.memType == MemOpType::MALLOC) {
        hostMemAllocation_[addr] = size;
        hostMemUsage_ = Utility::GetAddResult(hostMemUsage_, size);
    } else {
        if (hostMemAllocation_.find(addr) != hostMemAllocation_.end()) {
            hostMemUsage_ = Utility::GetSubResult(hostMemUsage_, hostMemAllocation_[addr]);
            hostMemAllocation_.erase(addr);
        } else {
            Utility::LogWarn("Invalid free addr %llx.", addr);
            return;
        }
    }

    JsonBaseInfo baseInfo{"memory", memRecord.pid, memRecord.tid, memRecord.kernelIndex};
    str = FormatCounterEvent(baseInfo, std::to_string(hostMemUsage_));
    return;
}

void TraceRecord::MemRecordToString(MemOpRecord &memRecord, std::string &str)
{
    if (memRecord.devType == DeviceType::NPU) {
        NpuMemRecordToString(memRecord, str);
    } else {
        CpuMemRecordToString(memRecord, str);
    }

    truePids_[Device{memRecord.devType, memRecord.devId}].insert(memRecord.pid);
    return;
}

void TraceRecord::KernelLaunchRecordToString(const KernelLaunchRecord &kernelLaunchRecord, std::string &str)
{
    JsonBaseInfo baseInfo{
        "kernel_" + std::to_string(kernelLaunchRecord.kernelLaunchIndex),
        kernelLaunchRecord.pid,
        kernelLaunchRecord.tid,
        kernelLaunchRecord.kernelLaunchIndex
    };
    str = FormatInstantEvent(baseInfo);
    return;
}

void TraceRecord::AclItfRecordToString(const AclItfRecord &aclItfRecord, std::string &str)
{
    if (aclItfRecord.devId == GD_INVALID_NUM) {
        return;
    }
    std::string name = aclItfRecord.type == AclOpType::INIT ? "acl_init" : "acl_finalize";
    JsonBaseInfo baseInfo{
        name,
        aclItfRecord.pid,
        aclItfRecord.tid,
        aclItfRecord.kernelIndex
    };
    str = FormatInstantEvent(baseInfo);
    return;
}

void TraceRecord::TorchRecordToString(const TorchNpuRecord &torchNpuRecord, std::string &str)
{
    MemoryUsage memoryUsage = torchNpuRecord.memoryUsage;
    uint64_t kernelIndex = torchNpuRecord.kernelIndex;
    uint64_t pid = torchNpuRecord.pid;
    uint64_t tid = torchNpuRecord.tid;

    truePids_[Device{DeviceType::NPU, torchNpuRecord.devId}].insert(pid);
    JsonBaseInfo reservedBaseInfo{"torch reserved memory", pid, tid, kernelIndex};
    JsonBaseInfo allocatedBaseInfo{"torch allocated memory", pid, tid, kernelIndex};

    str = FormatCounterEvent(reservedBaseInfo, std::to_string(memoryUsage.totalReserved));
    str += FormatCounterEvent(allocatedBaseInfo, std::to_string(memoryUsage.totalAllocated));

    return;
}

void TraceRecord::MstxRecordToString(const MstxRecord &mstxRecord, std::string &str)
{
    int32_t devId = mstxRecord.devId;
    std::string mstxEventName;

    std::lock_guard<std::mutex> lock(stepStartIndexMutex_);
    if (mstxRecord.markType == MarkType::MARK_A) {
        mstxEventName = "mstx_mark";
    } else if (mstxRecord.markType == MarkType::RANGE_START_A) {
        if (strcmp(mstxRecord.markMessage, "step start") == 0) {
            mstxEventName = "mstx_step" + std::to_string(mstxRecord.stepId) + "_start";
            stepStartIndex_[devId][mstxRecord.rangeId] = mstxRecord.kernelIndex;
        } else {
            mstxEventName = "mstx_range" + std::to_string(mstxRecord.rangeId) + "_start";
        }
    } else {
        if (stepStartIndex_.find(devId) == stepStartIndex_.end() ||
            stepStartIndex_[devId].find(mstxRecord.rangeId) == stepStartIndex_[devId].end()) {
            mstxEventName = "mstx_range" + std::to_string(mstxRecord.rangeId) + "_end";
        } else {
            mstxEventName = "mstx_step" + std::to_string(mstxRecord.stepId) + "_end";
            JsonBaseInfo stepBaseInfo{
                "step " + std::to_string(mstxRecord.stepId),
                mstxEventPid_,
                mstxRecord.tid,
                stepStartIndex_[devId][mstxRecord.rangeId]
            };
            str = FormatCompleteEvent(stepBaseInfo,
                                      mstxRecord.kernelIndex - stepStartIndex_[devId][mstxRecord.rangeId]);
        }
    }

    JsonBaseInfo baseInfo{
        mstxEventName,
        mstxEventPid_,
        mstxRecord.tid,
        mstxRecord.kernelIndex
    };
    str += FormatInstantEvent(baseInfo, mstxRecord.markMessage);
    return;
}

void TraceRecord::SetMetadataEvent(const Device &device)
{
    std::string str;
    uint64_t sortIndex = 0;

    if (truePids_.find(device) != truePids_.end()) {
        for (auto pid : truePids_[device]) {
            JsonBaseInfo sortBaseInfo{"process_sort_index", pid, 0, 0};
            str += FormatMetadataEvent(sortBaseInfo, "\"sort_index\": " + std::to_string(sortIndex++));
        }
    }

    for (auto eventPid : eventPids_) {
        JsonBaseInfo nameBaseInfo{"process_name", eventPid.pid, 0, 0};
        str += FormatMetadataEvent(nameBaseInfo, "\"name\": \"" + eventPid.name+ "\"");

        JsonBaseInfo sortBaseInfo{"process_sort_index", eventPid.pid, 0, 0};
        str += FormatMetadataEvent(sortBaseInfo, "\"sort_index\": " + std::to_string(sortIndex++));
    }

    SafeWriteString(str, device);
}

// TraceRecord生命周期结束时，文件写入完毕，关闭文件
TraceRecord::~TraceRecord()
{
    for (auto &file : traceFiles_) {
        FILE *fp = file.second.fp;
        if (fp != nullptr) {
            try {
                SetMetadataEvent(file.first);
                fprintf(fp, "{\n}\n]");
            } catch (const std::exception &ex) {
                std::cerr << "SetMetadataEvent fail, " << ex.what();
            }
            fclose(fp);
        }
    }
}
}