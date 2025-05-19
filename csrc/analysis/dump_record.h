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
    static DumpRecord& GetInstance(Config config);
    bool DumpData(const ClientId &clientId, const Record &record);
private:
    explicit DumpRecord(Config config);
    ~DumpRecord();
    void SetDirPath();
    DumpRecord(const DumpRecord&) = delete;
    DumpRecord& operator=(const DumpRecord&) = delete;
    DumpRecord(DumpRecord&& other) = delete;
    DumpRecord& operator=(DumpRecord&& other) = delete;

    bool WriteToFile(const DumpContainer &container, const CallStackString &stack);
    bool DumpMemData(const ClientId &clientId, const MemOpRecord &memrecord, const CallStackString &stack);
    bool DumpKernelData(const ClientId &clientId, const KernelLaunchRecord &kernelLaunchRecord);
    bool DumpAclItfData(const ClientId &clientId, const AclItfRecord &aclItfRecord);
    bool DumpMstxData(const ClientId &clientId, const MstxRecord &msxtRecord, const CallStackString &stack);
    bool DumpMemPoolData(const ClientId &clientId, const EventRecord &eventRecord, const CallStackString &stack);
    bool DumpAtbOpData(const ClientId &clientId, const AtbOpExecuteRecord &atbOpExecuteRecord);
    bool DumpAtbKernelData(const ClientId &clientId, const AtbKernelRecord &atbKernelRecord);
    bool DumpAtenOpLaunchData(const ClientId &clientId, const AtenOpLaunchRecord &atenOpLaunchRecord,
    const CallStackString &stack);
    bool DumpMemAccessData(const ClientId &clientId, const MemAccessRecord &memAccessRecord,
    const CallStackString &stack);
    FILE *leaksDataFile_ = nullptr;
    std::unordered_map<ClientId, std::unordered_map<uint64_t, uint64_t>> hostMemSizeMap_;
    std::unordered_map<ClientId, std::unordered_map<uint64_t, uint64_t>> memSizeMap_;
    std::unordered_map<ClientId, std::unordered_map<uint64_t, MemOpSpace>> memOpMap_;
    std::unordered_map<ClientId, uint64_t> memHost_;
    std::unordered_map<ClientId, uint64_t> memDevice_;
    std::string dirPath_;
    std::mutex fileMutex_;
    std::string fileNamePrefix_ = "leaks_dump_";
    std::string csvHeader_;
    Config config_;
};

template <typename T>
void CopyMemPoolRecordMember(const T &record, DumpContainer &container)
{
    container.id = record.recordIndex;
    container.pid = record.pid;
    container.tid = record.tid;
    container.timeStamp = record.timeStamp;
    container.deviceId = std::to_string(record.devId);
}

} // namespace Leaks

#endif