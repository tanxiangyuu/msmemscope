// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 
#ifndef TRACE_RECORD_H
#define TRACE_RECORD_H
 
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <mutex>
#include "framework/record_info.h"
#include "framework/config_info.h"
#include "host_injection/core/Communication.h"

namespace Leaks {

struct JsonBaseInfo {
    std::string name;
    uint64_t pid;
    uint64_t tid;
    uint64_t ts;
};

struct TorchMemLeakInfo {
    int32_t devId;
    uint64_t kernelIndex;
    uint64_t duration;
    uint64_t addr;
    int64_t size;
};

struct FileInfo {
    FILE* fp;
    std::string filePath;
};

struct MemAllocationInfo {
    uint64_t size;
    int32_t devId;
};

struct EventPid {
    uint64_t pid;
    std::string name;
};

struct Device {
    DeviceType type;
    int32_t index;
    bool operator==(const Device& other) const
    {
        return type == other.type && index == other.index;
    }
};

struct DeviceStructHash {
    size_t operator()(const Device& d) const
    {
        size_t hash1 = std::hash<uint8_t>{}(static_cast<uint8_t>(d.type));
        size_t hash2 = std::hash<int32_t>{}(d.index);
        return hash1 ^ (hash2 << 1);
    }
};

class TraceRecord {
public:
    static TraceRecord& GetInstance();
    void TraceHandler(const EventRecord &record);
    void ProcessTorchMemLeakInfo(const TorchMemLeakInfo &info);

    std::unordered_map<Device, std::unordered_set<uint64_t>, DeviceStructHash> truePids_;
    std::unordered_map<Device, FileInfo, DeviceStructHash> traceFiles_;

private:
    TraceRecord();
    ~TraceRecord();
    TraceRecord(const TraceRecord&) = delete;
    TraceRecord& operator=(const TraceRecord&) = delete;
    TraceRecord(TraceRecord&& other) = delete;
    TraceRecord& operator=(TraceRecord&& other) = delete;

    void SetDirPath();

    void MemRecordToString(MemOpRecord &memRecord, std::string &str);
    void NpuMemRecordToString(MemOpRecord &memRecord, std::string &str);
    void CpuMemRecordToString(const MemOpRecord &memRecord, std::string &str);
    void KernelLaunchRecordToString(const KernelLaunchRecord &kernelLaunchRecord, std::string &str);
    void SaveKernelLaunchRecordToCpuTrace(const std::string &str);
    void AclItfRecordToString(const AclItfRecord &aclItfRecord, std::string &str);
    void TorchRecordToString(const TorchNpuRecord &torchNpuRecord, std::string &str);
    void MstxRecordToString(const MstxRecord &mstxRecord, std::string &str);
    void TorchMemLeakInfoToString(const TorchMemLeakInfo &info, std::string &str);

    bool CheckStrHasContent(const std::string &str);
    bool CheckFileExistByDevice(const Device &device);
    void SafeWriteString(const std::string &str, const Device &device);
    bool CreateFileByDevice(const Device &device);

    void ProcessRecord(const EventRecord &record);
    void SetMetadataEvent(const Device &device);

    uint64_t mstxEventPid_ = 0;
    uint64_t leakEventPid_ = 1;
    std::vector<EventPid> eventPids_;

    std::mutex fileMutex_;
    std::string dirPath_;
    std::unordered_map<Device, std::mutex, DeviceStructHash> writeFileMutex_;

    std::mutex halMemMutex_;
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, MemAllocationInfo>> halDeviceMemAllocation_;
    std::unordered_map<uint64_t, std::unordered_map<int32_t, uint64_t>> halDeviceMemUsage_;
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, MemAllocationInfo>> halHostMemAllocation_;
    std::unordered_map<uint64_t, std::unordered_map<int32_t, uint64_t>> halHostMemUsage_;

    std::mutex hostMemMutex_;
    std::unordered_map<uint64_t, uint64_t> hostMemAllocation_;
    uint64_t hostMemUsage_ = 0;

    std::mutex stepStartIndexMutex_;
    std::unordered_map<int32_t, std::unordered_map<uint64_t, uint64_t>> stepStartIndex_;
};
}
#endif