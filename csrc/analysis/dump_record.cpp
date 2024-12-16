// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include "log.h"
#include "dump_record.h"

namespace Leaks {
constexpr uint32_t DIRMOD = 0777;
bool DumpRecord::CreateFile(const ClientId &clientId, FILE *clientfp, std::string type)
{
    std::string dirPath = "leaksDumpResults";
    if (access(dirPath.c_str(), F_OK) == -1) {
        Utility::LogInfo("dir %s does not exist", dirPath.c_str());
        if (mkdir(dirPath.c_str(), DIRMOD) != 0) {
            Utility::LogError("cannot create dir %s", dirPath.c_str());
            return false;
        }
    }
    if (clientfp == nullptr) {
        auto now = std::chrono::system_clock::now();
        std::time_t time = std::chrono::system_clock::to_time_t(now);
        std::string filePath = dirPath + "/" + std::to_string(clientId) + type + std::to_string(time) + ".csv";
        FILE* fp = fopen(filePath.c_str(), "a");
        if (fp != nullptr) {
            if (type == "torchnpu") {
                fprintf(fp, "device_type, device_index, data_type, allocator_type, ptr, recordIndex, alloc_size, \
total_allocated, total_reserved, total_active, stream_ptr\n");
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
    std::string space;
    uint64_t totalMem;
    if (memRecord.memType == MemOpType::MALLOC) {
        if (memRecord.space == MemOpSpace::HOST) {
            memHost[clientId] += memRecord.memSize;
            memSizeMap[clientId][memRecord.addr] += memRecord.memSize;
            memOpMap[clientId][memRecord.addr] = MemOpSpace::HOST;
            space = "host";
            totalMem = memHost[clientId];
        } else if (memRecord.space == MemOpSpace::DEVICE) {
            memDevice[clientId] += memRecord.memSize;
            memSizeMap[clientId][memRecord.addr] += memRecord.memSize;
            memOpMap[clientId][memRecord.addr] = MemOpSpace::DEVICE;
            space = "device";
            totalMem = memDevice[clientId];
        }
    } else {
        if (memOpMap[clientId][memRecord.addr] == MemOpSpace::HOST) {
            memHost[clientId] -= memSizeMap[clientId][memRecord.addr];
            space = "host";
            totalMem = memHost[clientId];
        } else if (memOpMap[clientId][memRecord.addr] == MemOpSpace::DEVICE) {
            memDevice[clientId] -= memSizeMap[clientId][memRecord.addr];
            space = "device";
            totalMem = memDevice[clientId];
        }
        memOpMap[clientId][memRecord.addr] = MemOpSpace::INVALID;
        memSizeMap[clientId][memRecord.addr] = 0;
    }
    std::string memOp = memRecord.memType == MemOpType::MALLOC ? "malloc" : "free";

    fprintf(leaksDataFile[clientId], "%s,%lu,%lu,%lu,%d,%lu,%lu,%lu,%llu,%d,%s,%lu,%lu,%lu\n",
            memOp.c_str(), memRecord.pid, memRecord.tid, clientId, memRecord.devid, memRecord.recordIndex,
            memRecord.timeStamp, memRecord.kernelIndex, memRecord.flag, memRecord.modid, space.c_str(),
            memRecord.addr, memSizeMap[clientId][memRecord.addr], totalMem);
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
        aclItfRecord.aclItfRecord);
    return true;
}
bool DumpRecord::DumpTorchData(const ClientId &clientId, const TorchNpuRecord &torchNpuRecord)
{
    if (!CreateFile(clientId, torchNpuDataFile[clientId], "torchnpu")) {
        return false;
    }
    MemoryUsage memoryUsage = torchNpuRecord.memoryUsage;
    fprintf(torchNpuDataFile[clientId], "%d,%d,%d,%d,%ld,%lu,%ld,%ld,%ld,%ld,%ld\n",
        memoryUsage.device_type, memoryUsage.device_index, memoryUsage.data_type,
        memoryUsage.allocator_type, memoryUsage.ptr, torchNpuRecord.recordIndex, memoryUsage.alloc_size,
        memoryUsage.total_allocated, memoryUsage.total_reserved, memoryUsage.total_active, memoryUsage.stream_ptr);
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