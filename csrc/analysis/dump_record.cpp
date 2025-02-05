// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "dump_record.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include "log.h"
#include "file.h"
#include "utils.h"

namespace Leaks {
constexpr uint32_t DIRMOD = 0777;

DumpRecord& DumpRecord::GetInstance()
{
    static DumpRecord instance;
    return instance;
}

bool DumpRecord::DumpData(const ClientId &clientId, const EventRecord &record)
{
    fileName = "leaks" + Utility::GetDateStr() + ".csv";
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
        case RecordType::MSTX_MARK_RECORD: {
            auto mstxRecord = record.record.mstxRecord;
            if (!DumpMstxData(clientId, mstxRecord)) {
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
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile, dirPath, fileName, headers)) {
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
    fprintf(leaksDataFile, "%s,N/A,%lu,%lu,%lu,%d,%lu,%lu,%lu,%llu,%d,%d,%lu,%lu,%lu,"
            "N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A\n",
            memOp.c_str(), memRecord.pid, memRecord.tid, clientId, memRecord.devId, memRecord.recordIndex,
            memRecord.timeStamp, memRecord.kernelIndex, memRecord.flag, memRecord.modid, int(space),
            memRecord.addr, currentSize, totalMem);
    return true;
}
bool DumpRecord::DumpKernelData(const ClientId &clientId, const KernelLaunchRecord &kernelLaunchRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile, dirPath, fileName, headers)) {
        return false;
    }
    std::string name;
    if (kernelLaunchRecord.kernelName[0] == '\0') {
        name = "N/A";
    } else {
        name = kernelLaunchRecord.kernelName;
    }
    fprintf(leaksDataFile, "kernelLaunch,%s,%lu,%lu,%lu,%lu,%lu,%lu,%lu,N/A,N/A,N/A,N/A,N/A,N/A,"
            "N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A\n", name.c_str(),
            kernelLaunchRecord.pid, kernelLaunchRecord.tid, clientId, kernelLaunchRecord.devId,
            kernelLaunchRecord.recordIndex, kernelLaunchRecord.timeStamp, kernelLaunchRecord.kernelLaunchIndex);
    return true;
}

bool DumpRecord::DumpMstxData(const ClientId &clientId, const MstxRecord &mstxRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile, dirPath, fileName, headers)) {
        return false;
    }
    std::string name;
    switch (mstxRecord.markType) {
        case Leaks::MarkType::MARK_A: {
            name = "Mark";
            break;
        }
        case Leaks::MarkType::RANGE_START_A: {
            name = "Range_start";
            break;
        }
        case Leaks::MarkType::RANGE_END: {
            name = "Range_end";
            break;
        }
        default: {
            name = "N/A";
            break;
        }
    }
    fprintf(leaksDataFile, "mstx,%s,%lu,%lu,%lu,%lu,%lu,%lu,N/A,N/A,N/A,N/A,N/A,N/A,N/A,"
            "N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A\n", name.c_str(), mstxRecord.pid, mstxRecord.tid, clientId,
            mstxRecord.devId, mstxRecord.recordIndex, mstxRecord.timeStamp);
    return true;
}

bool DumpRecord::DumpAclItfData(const ClientId &clientId, const AclItfRecord &aclItfRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile, dirPath, fileName, headers)) {
        return false;
    }
    fprintf(leaksDataFile, "aclItfRecord,N/A,%lu,%lu,%lu,N/A,%lu,%lu,%lu,N/A,N/A,N/A,N/A,N/A,N/A,"
            "N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A\n", aclItfRecord.pid, aclItfRecord.tid, clientId,
            aclItfRecord.recordIndex, aclItfRecord.timeStamp, aclItfRecord.aclItfRecordIndex);
    return true;
}
bool DumpRecord::DumpTorchData(const ClientId &clientId, const TorchNpuRecord &torchNpuRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile, dirPath, fileName, headers)) {
        return false;
    }
    MemoryUsage memoryUsage = torchNpuRecord.memoryUsage;
    fprintf(leaksDataFile, "torch_npu,N/A,%lu,%lu,%lu,%lu,%lu,%lu,N/A,N/A,N/A,N/A,N/A,N/A,N/A,"
            "%d,%d,%d,%d,%ld,%ld,%ld,%ld,%ld,%ld\n", torchNpuRecord.pid, torchNpuRecord.tid, clientId,
            torchNpuRecord.devId, torchNpuRecord.recordIndex, torchNpuRecord.timeStamp, memoryUsage.deviceType,
            memoryUsage.deviceIndex, memoryUsage.dataType, memoryUsage.allocatorType, memoryUsage.ptr,
            memoryUsage.allocSize, memoryUsage.totalAllocated, memoryUsage.totalReserved, memoryUsage.totalActive,
            memoryUsage.streamPtr);
    return true;
}

DumpRecord::~DumpRecord()
{
    if (leaksDataFile != nullptr) {
        fclose(leaksDataFile);
    }
}
}