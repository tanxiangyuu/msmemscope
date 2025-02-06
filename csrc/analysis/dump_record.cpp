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
    if (!Utility::CreateCsvFile(&leaksDataFile_, dirPath_, fileNamePrefix_, headers_)) {
        return false;
    }
    MemOpSpace space;
    uint64_t currentSize;
    if (memRecord.memType == MemOpType::MALLOC) {
        memSizeMap_[clientId][memRecord.addr] = memRecord.memSize;
        memOpMap_[clientId][memRecord.addr] = memRecord.space;
        space = memRecord.space;
        currentSize = memRecord.memSize;
        if (space == MemOpSpace::HOST) {
            memHost_[clientId] = Utility::GetAddResult(memHost_[clientId], currentSize);
        } else if (space == MemOpSpace::DEVICE) {
            memDevice_[clientId] = Utility::GetAddResult(memDevice_[clientId], currentSize);
        }
    } else {
        currentSize = memSizeMap_[clientId][memRecord.addr];
        space = memOpMap_[clientId][memRecord.addr];
        if (space == MemOpSpace::HOST) {
            memHost_[clientId] = Utility::GetSubResult(memHost_[clientId], currentSize);
        } else if (space == MemOpSpace::DEVICE) {
            memDevice_[clientId] = Utility::GetSubResult(memDevice_[clientId], currentSize);
        }
        memOpMap_[clientId][memRecord.addr] = MemOpSpace::INVALID;
        memSizeMap_[clientId][memRecord.addr] = 0;
    }
    std::string memOp = memRecord.memType == MemOpType::MALLOC ? "malloc" : "free";

    uint64_t totalMem = space == MemOpSpace::HOST ? memHost_[clientId] : memDevice_[clientId];
    fprintf(leaksDataFile_, "%s,N/A,%lu,%lu,%lu,%d,%lu,%lu,%lu,%llu,%d,%d,%lu,%lu,%lu,"
            "N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A\n",
            memOp.c_str(), memRecord.pid, memRecord.tid, clientId, memRecord.devId, memRecord.recordIndex,
            memRecord.timeStamp, memRecord.kernelIndex, memRecord.flag, memRecord.modid, int(space),
            memRecord.addr, currentSize, totalMem);
    return true;
}
bool DumpRecord::DumpKernelData(const ClientId &clientId, const KernelLaunchRecord &kernelLaunchRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile_, dirPath_, fileNamePrefix_, headers_)) {
        return false;
    }
    std::string name;
    if (kernelLaunchRecord.kernelName[0] == '\0') {
        name = "N/A";
    } else {
        name = kernelLaunchRecord.kernelName;
    }
    fprintf(leaksDataFile_, "kernelLaunch,%s,%lu,%lu,%lu,%d,%lu,%lu,%lu,N/A,N/A,N/A,N/A,N/A,N/A,"
            "N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A\n", name.c_str(),
            kernelLaunchRecord.pid, kernelLaunchRecord.tid, clientId, kernelLaunchRecord.devId,
            kernelLaunchRecord.recordIndex, kernelLaunchRecord.timeStamp, kernelLaunchRecord.kernelLaunchIndex);
    return true;
}

bool DumpRecord::DumpMstxData(const ClientId &clientId, const MstxRecord &mstxRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile_, dirPath_, fileNamePrefix_, headers_)) {
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
    fprintf(leaksDataFile_, "mstx,%s,%lu,%lu,%lu,%d,%lu,%lu,N/A,N/A,N/A,N/A,N/A,N/A,N/A,"
            "N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A\n", name.c_str(), mstxRecord.pid, mstxRecord.tid, clientId,
            mstxRecord.devId, mstxRecord.recordIndex, mstxRecord.timeStamp);
    return true;
}

bool DumpRecord::DumpAclItfData(const ClientId &clientId, const AclItfRecord &aclItfRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile_, dirPath_, fileNamePrefix_, headers_)) {
        return false;
    }
    fprintf(leaksDataFile_, "aclItfRecord,N/A,%lu,%lu,%lu,%d,%lu,%lu,%lu,N/A,N/A,N/A,N/A,N/A,N/A,"
            "N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A\n", aclItfRecord.pid, aclItfRecord.tid, clientId,
            aclItfRecord.devId, aclItfRecord.recordIndex, aclItfRecord.timeStamp, aclItfRecord.aclItfRecordIndex);
    return true;
}
bool DumpRecord::DumpTorchData(const ClientId &clientId, const TorchNpuRecord &torchNpuRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile_, dirPath_, fileNamePrefix_, headers_)) {
        return false;
    }
    MemoryUsage memoryUsage = torchNpuRecord.memoryUsage;
    fprintf(leaksDataFile_, "torch_npu,N/A,%lu,%lu,%lu,%d,%lu,%lu,N/A,N/A,N/A,N/A,N/A,N/A,N/A,"
            "%d,%d,%d,%d,%ld,%ld,%ld,%ld,%ld,%ld\n", torchNpuRecord.pid, torchNpuRecord.tid, clientId,
            torchNpuRecord.devId, torchNpuRecord.recordIndex, torchNpuRecord.timeStamp, memoryUsage.deviceType,
            memoryUsage.deviceIndex, memoryUsage.dataType, memoryUsage.allocatorType, memoryUsage.ptr,
            memoryUsage.allocSize, memoryUsage.totalAllocated, memoryUsage.totalReserved, memoryUsage.totalActive,
            memoryUsage.streamPtr);
    return true;
}

DumpRecord::~DumpRecord()
{
    if (leaksDataFile_ != nullptr) {
        fclose(leaksDataFile_);
    }
}
}