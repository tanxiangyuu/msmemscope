// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef DUMP_RECORD_H
#define DUMP_RECORD_H

#include <unordered_map>
#include <string>
#include <cstdio>
#include <mutex>
#include "framework/record_info.h"
#include "framework/config_info.h"
#include "host_injection/core/Communication.h"

namespace Leaks {

// DumpRecord类主要用于将analyzer分析的数据dump至csv文件
class DumpRecord {
public:
    static DumpRecord& GetInstance();
    bool DumpData(const ClientId &clientId, const EventRecord &record);
private:
    DumpRecord();
    ~DumpRecord();
    void SetDirPath();
    DumpRecord(const DumpRecord&) = delete;
    DumpRecord& operator=(const DumpRecord&) = delete;
    DumpRecord(DumpRecord&& other) = delete;
    DumpRecord& operator=(DumpRecord&& other) = delete;

    bool DumpMemData(const ClientId &clientId, const MemOpRecord &memrecord);
    bool DumpKernelData(const ClientId &clientId, const KernelLaunchRecord &kernelLaunchRecord);
    bool DumpAclItfData(const ClientId &clientId, const AclItfRecord &aclItfRecord);
    bool DumpTorchData(const ClientId &clientId, const TorchNpuRecord &torchNpuRecord);
    bool DumpMstxData(const ClientId &clientId, const MstxRecord &msxtRecord);
    FILE *leaksDataFile_ = nullptr;
    std::unordered_map<ClientId, std::unordered_map<uint64_t, uint64_t>> memSizeMap_;
    std::unordered_map<ClientId, std::unordered_map<uint64_t, MemOpSpace>> memOpMap_;
    std::unordered_map<ClientId, uint64_t> memHost_;
    std::unordered_map<ClientId, uint64_t> memDevice_;
    std::string dirPath_;
    std::mutex fileMutex_;
    std::string fileNamePrefix_ = "leaks_dump_";
};
}
#endif