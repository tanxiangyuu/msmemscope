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
#include "device_manager.h"
#include "data_handler.h"

namespace Leaks {

// DumpRecord类主要用于将analyzer分析的数据dump至csv文件
class DumpRecord {
public:
    static DumpRecord& GetInstance(Config config);
    bool DumpData(const ClientId &clientId, const RecordBase *record);
    bool WriteToFile(DumpContainer &container, const CallStackString &stack);
    void SetAllocAttr(MemStateInfo& memInfo);
private:
    explicit DumpRecord(Config config);
    ~DumpRecord() = default;
    DumpRecord(const DumpRecord&) = delete;
    DumpRecord& operator=(const DumpRecord&) = delete;
    DumpRecord(DumpRecord&& other) = delete;
    DumpRecord& operator=(DumpRecord&& other) = delete;

    bool DumpMemData(const ClientId &clientId, const MemOpRecord *memrecord);
    bool DumpKernelData(const ClientId &clientId, const KernelLaunchRecord *kernelLaunchRecord);
    bool DumpKernelExcuteData(const KernelExcuteRecord *record);
    bool DumpAclItfData(const ClientId &clientId, const AclItfRecord *aclItfRecord);
    bool DumpMstxData(const ClientId &clientId, const MstxRecord *msxtRecord);
    bool DumpMemPoolData(const ClientId &clientId, const MemPoolRecord *memPoolRecord);
    bool DumpAtbOpData(const ClientId &clientId, const AtbOpExecuteRecord *atbOpExecuteRecord);
    bool DumpAtbKernelData(const ClientId &clientId, const AtbKernelRecord *atbKernelRecord);
    bool DumpAtenOpLaunchData(const ClientId &clientId, const AtenOpLaunchRecord *atenOpLaunchRecord);
    std::mutex fileMutex_;
    Config config_;
    std::unique_ptr<DataHandler> handler_;
};

} // namespace Leaks

#endif