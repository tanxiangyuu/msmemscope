// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 
#ifndef TRACE_RECORD_H
#define TRACE_RECORD_H
 
#include <unordered_map>
#include <string>
#include <cstdio>
#include <cstdint>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <mutex>
#include "framework/record_info.h"
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
    uint64_t timestamp;
    uint64_t duration;
    uint64_t addr;
    int64_t size;
};

struct FileInfo {
    FILE* fp;
    std::string filePath;
};

class TraceRecord {
public:
    static TraceRecord& GetInstance();
    void TraceHandler(const EventRecord &record);
    bool ProcessRecord(const EventRecord &record);
    void ProcessTorchMemLeakInfo(const TorchMemLeakInfo &info);

    std::unordered_map<int32_t, FileInfo> traceFiles_;

private:
    TraceRecord(){};
    ~TraceRecord();
    TraceRecord(const TraceRecord&) = delete;
    TraceRecord& operator=(const TraceRecord&) = delete;
    TraceRecord(TraceRecord&& other) = delete;
    TraceRecord& operator=(TraceRecord&& other) = delete;

    void RecordToString(const MemOpRecord &memrecord, std::string &str);
    void RecordToString(const KernelLaunchRecord &kernelLaunchRecord, std::string &str);
    void RecordToString(const AclItfRecord &aclItfRecord, std::string &str);
    void RecordToString(const TorchNpuRecord &torchNpuRecord, std::string &str);
    void RecordToString(const MstxRecord &mstxRecord, std::string &str);
    void TorchMemLeakInfoToString(const TorchMemLeakInfo &info, std::string &str);
    bool CheckStrHasContent(const std::string &str);
    bool CheckFileExistWithDeviceId(const int32_t &devId);
    void SafeWriteString(const std::string &str, const int32_t &devId);
    bool CreateFileWithDeviceId(const int32_t &devId);
    
    std::mutex createFileMutex_;
    std::unordered_map<int32_t, std::mutex> writeFileMutex_;
    
    std::unordered_map<int32_t, std::unordered_map<uint64_t, uint64_t>> hostMemAllocation_;
    std::unordered_map<int32_t, std::unordered_map<uint64_t, uint64_t>> deviceMemAllocation_;
    std::unordered_map<int32_t, uint64_t> hostMemUsage_;
    std::unordered_map<int32_t, uint64_t> deviceMemUsage_;

    std::unordered_map<int32_t, uint64_t> lastStepStartTime_;
};
}
#endif