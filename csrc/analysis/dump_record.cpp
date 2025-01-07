// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "dump_record.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <climits>
#include "log.h"
#include "file.h"
#include "utils.h"

namespace Leaks {
constexpr uint32_t DIRMOD = 0777;

bool DumpRecord::CreateFile(const ClientId &clientId, FILE *clientfp, std::string type)
{
    std::string dirPath = "leaksDumpResults";
    if (!Utility::MakeDir(dirPath)) {
        return false;
    }
    if (clientfp == nullptr) {
        auto now = std::chrono::system_clock::now();
        std::time_t time = std::chrono::system_clock::to_time_t(now);
        std::string filePath = dirPath + "/" + std::to_string(clientId) + type + std::to_string(time) + ".csv";
        FILE* fp = fopen(filePath.c_str(), "a");
        if (fp != nullptr) {
            if (type == "torchnpu") {
                fprintf(fp, "deviceType, deviceIndex, dataType, allocatorType, ptr, recordIndex, allocSize, \
totalAllocated, totalReserved, totalActive, streamPtr\n");
                torchNpuDataFile[clientId] = fp;
            } else {
                fprintf(fp, "type, processID, threadID, clientID, deviceID, recordIndex, timeStamp, \
kernelIndex, flag, moduleID, host/device, addr, size, sumMemory\n");
                leaksDataFile[clientId] = fp;
            }
        } else {
            Utility::LogError("clientId %d open file %s error", clientId, filePath.c_str());
            return false;
        }
    }
    return true;
}
bool DumpRecord::DumpData(const ClientId &clientId, const EventRecord &record)
{
    switch (record.type) {
        case RecordType::MEMORY_RECORD: {
            auto memRecord = record.record.memoryRecord;
            if (!DumpMemData(clientId, memRecord)) {
                return false;
            }
            break;
        }
        case RecordType::KERNEL_LAUNCH_RECORD: {
            auto kernelLaunchRecord = record.record.kernelLaunchRecord;
            if (!DumpKernelData(clientId, kernelLaunchRecord)) {
                return false;
            }
            break;
        }
        case RecordType::ACL_ITF_RECORD: {
            auto aclItfRecord = record.record.aclItfRecord;
            if (!DumpAclItfData(clientId, aclItfRecord)) {
                return false;
            }
            break;
        }
        case RecordType::TORCH_NPU_RECORD: {
            TorchNpuRecord torchNpuRecord = record.record.torchNpuRecord;
            if (!DumpTorchData(clientId, torchNpuRecord)) {
                return false;
            }
            break;
        }
        default:
            break;
    }
    return true;
}
bool DumpRecord::DumpMemData(const ClientId &clientId, const MemOpRecord &memRecord)
{
    if (!CreateFile(clientId, leaksDataFile[clientId], "msleaks")) {
        return false;
    }
    MemOpSpace space;
    uint64_t currentSize;
    if (memRecord.memType == MemOpType::MALLOC) {
        memSizeMap[clientId][memRecord.addr] = memRecord.memSize;
        memOpMap[clientId][memRecord.addr] = memRecord.space;
        space = memRecord.space;
        currentSize = memRecord.memSize;
        if (space == MemOpSpace::HOST) {
            memHost[clientId] = Utility::GetAddResult(memHost[clientId], currentSize);
        } else if (space == MemOpSpace::DEVICE) {
            memDevice[clientId] = Utility::GetAddResult(memDevice[clientId], currentSize);
        }
    } else {
        currentSize = memSizeMap[clientId][memRecord.addr];
        space = memOpMap[clientId][memRecord.addr];
        if (space == MemOpSpace::HOST) {
            memHost[clientId] = Utility::GetSubResult(memHost[clientId], currentSize);
        } else if (space == MemOpSpace::DEVICE) {
            memDevice[clientId] = Utility::GetSubResult(memDevice[clientId], currentSize);
        }
        memOpMap[clientId][memRecord.addr] = MemOpSpace::INVALID;
        memSizeMap[clientId][memRecord.addr] = 0;
    }
    std::string memOp = memRecord.memType == MemOpType::MALLOC ? "malloc" : "free";

    uint64_t totalMem = space == MemOpSpace::HOST ? memHost[clientId] : memDevice[clientId];
    fprintf(leaksDataFile[clientId], "%s,%lu,%lu,%lu,%d,%lu,%lu,%lu,%llu,%d,%d,%lu,%lu,%lu\n",
            memOp.c_str(), memRecord.pid, memRecord.tid, clientId, memRecord.devId, memRecord.recordIndex,
            memRecord.timeStamp, memRecord.kernelIndex, memRecord.flag, memRecord.modid, int(space),
            memRecord.addr, currentSize, totalMem);
    return true;
}
bool DumpRecord::DumpKernelData(const ClientId &clientId, const KernelLaunchRecord &kernelLaunchRecord)
{
    if (!CreateFile(clientId, leaksDataFile[clientId], "msleaks")) {
        return false;
    }
    fprintf(leaksDataFile[clientId], "kernelLaunch,%lu,%lu,%lu,null,%lu,%lu,%lu,null,null,null,null,null,null\n",
        kernelLaunchRecord.pid, kernelLaunchRecord.tid, clientId, kernelLaunchRecord.recordIndex,
        kernelLaunchRecord.timeStamp, kernelLaunchRecord.kernelLaunchIndex);
    return true;
}
bool DumpRecord::DumpAclItfData(const ClientId &clientId, const AclItfRecord &aclItfRecord)
{
    if (!CreateFile(clientId, leaksDataFile[clientId], "msleaks")) {
        return false;
    }
    fprintf(leaksDataFile[clientId], "aclItfRecord,%lu,%lu,%lu,null,%lu,%lu,%lu,null,null,null,null,null,null\n",
        aclItfRecord.pid, aclItfRecord.tid, clientId, aclItfRecord.recordIndex, aclItfRecord.timeStamp,
        aclItfRecord.aclItfRecordIndex);
    return true;
}
bool DumpRecord::DumpTorchData(const ClientId &clientId, const TorchNpuRecord &torchNpuRecord)
{
    if (!CreateFile(clientId, torchNpuDataFile[clientId], "torchnpu")) {
        return false;
    }
    MemoryUsage memoryUsage = torchNpuRecord.memoryUsage;
    fprintf(torchNpuDataFile[clientId], "%d,%d,%d,%d,%ld,%lu,%ld,%ld,%ld,%ld,%ld\n",
        memoryUsage.deviceType, memoryUsage.deviceIndex, memoryUsage.dataType,
        memoryUsage.allocatorType, memoryUsage.ptr, torchNpuRecord.recordIndex, memoryUsage.allocSize,
        memoryUsage.totalAllocated, memoryUsage.totalReserved, memoryUsage.totalActive, memoryUsage.streamPtr);
    return true;
}
DumpRecord::DumpRecord()
{
}

DumpRecord::~DumpRecord()
{
    for (auto &p : leaksDataFile) {
        fclose(p.second);
    }
    for (auto &p : torchNpuDataFile) {
        fclose(p.second);
    }
}
}