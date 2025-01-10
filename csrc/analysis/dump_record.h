// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef DUMP_RECORD_H
#define DUMP_RECORD_H

#include <unordered_map>
#include <string>
#include <cstdio>
#include <mutex>
#include "framework/record_info.h"
#include "host_injection/core/Communication.h"

namespace Leaks {

// DumpRecord类主要用于将analyzer分析的数据dump至csv文件
class DumpRecord {
public:
    DumpRecord();
    bool DumpData(const ClientId &clientId, const EventRecord &record);
    ~DumpRecord();
private:
    bool DumpMemData(const ClientId &clientId, const MemOpRecord &memrecord);
    bool DumpKernelData(const ClientId &clientId, const KernelLaunchRecord &kernelLaunchRecord);
    bool DumpAclItfData(const ClientId &clientId, const AclItfRecord &aclItfRecord);
    bool DumpTorchData(const ClientId &clientId, const TorchNpuRecord &torchNpuRecord);
    bool DumpMstxData(const ClientId &clientId, const MstxRecord &msxtRecord);
    FILE *leaksDataFile = nullptr;
    std::unordered_map<ClientId, std::unordered_map<uint64_t, uint64_t>> memSizeMap;
    std::unordered_map<ClientId, std::unordered_map<uint64_t, MemOpSpace>> memOpMap;
    std::unordered_map<ClientId, uint64_t> memHost;
    std::unordered_map<ClientId, uint64_t> memDevice;
    std::string headers = "type,name,processID,threadID,clientID,deviceID,recordIndex,timeStamp,"
        "kernelIndex,flag,moduleID,host/device,addr,size,sumMemory,device_type,device_index,data_type,"
        "allocator_type,ptr,alloc_size,total_allocated,total_reserved,total_active,stream_ptr\n";
    std::string dirPath = "leaksDumpResults";
    std::string fileName;
    std::mutex fileMutex_;
};
}
#endif