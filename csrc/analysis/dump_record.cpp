// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "dump_record.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include "log.h"
#include "file.h"
#include "utils.h"
#include "config_info.h"

namespace Leaks {

DumpRecord& DumpRecord::GetInstance()
{
    static DumpRecord instance;
    return instance;
}

DumpRecord::DumpRecord()
{
    SetDirPath();
}

void DumpRecord::SetDirPath()
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    dirPath_ = Utility::g_dirPath + "/" + std::string(DUMP_FILE);
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
        case RecordType::ATB_MEMORY_POOL_RECORD:
        case RecordType::TORCH_NPU_RECORD: {
            if (!DumpMemPoolData(clientId, record)) {
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
    if (!Utility::CreateCsvFile(&leaksDataFile_, dirPath_, fileNamePrefix_, LEAKS_HEADERS)) {
        return false;
    }
    uint64_t currentSize;
    if (memRecord.devType == DeviceType::CPU) {
        if (memRecord.memType == MemOpType::MALLOC) {
            hostMemSizeMap_[clientId][memRecord.addr] = memRecord.memSize;
            currentSize = memRecord.memSize;
        } else if (hostMemSizeMap_.find(clientId) != hostMemSizeMap_.end()
            && hostMemSizeMap_[clientId].find(memRecord.addr) != hostMemSizeMap_[clientId].end()) {
            currentSize = hostMemSizeMap_[clientId][memRecord.addr];
            hostMemSizeMap_[clientId].erase(memRecord.addr);
        } else {
            return false;
        }
    } else {
        if (memRecord.memType == MemOpType::MALLOC) {
            memSizeMap_[clientId][memRecord.addr] = memRecord.memSize;
            currentSize = memRecord.memSize;
        } else {
            currentSize = memSizeMap_[clientId][memRecord.addr];
            memSizeMap_[clientId][memRecord.addr] = 0;
        }
    }
    std::string memOp = memRecord.memType == MemOpType::MALLOC ? "malloc" : "free";
    std::string deviceType;
    if (memRecord.devId == GD_INVALID_NUM) {
        deviceType = "N/A";
    } else {
        deviceType = memRecord.space == MemOpSpace::HOST
                     || memRecord.devType == DeviceType::CPU ?
                     "host" : std::to_string(memRecord.devId);
    }
    int fpRes = fprintf(leaksDataFile_, "%lu,%lu,%s,N/A,%lu,%lu,%s,%lu,%llu,%lu,%lu,N/A,N/A\n",
                        memRecord.recordIndex, memRecord.timeStamp, memOp.c_str(), memRecord.pid, memRecord.tid,
                        deviceType.c_str(), memRecord.kernelIndex, memRecord.flag, memRecord.addr, currentSize);
    if (fpRes < 0) {
        std::cout << "[msleaks] Error: Fail to write data to csv file, errno:" << fpRes << std::endl;
        return false;
    }
    return true;
}
bool DumpRecord::DumpKernelData(const ClientId &clientId, const KernelLaunchRecord &kernelLaunchRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile_, dirPath_, fileNamePrefix_, LEAKS_HEADERS)) {
        return false;
    }
    std::string name;
    if (kernelLaunchRecord.kernelName[0] == '\0') {
        name = "N/A";
    } else {
        name = kernelLaunchRecord.kernelName;
    }
    int fpRes = fprintf(leaksDataFile_, "%lu,%lu,kernelLaunch,%s,%lu,%lu,%d,%lu,N/A,N/A,N/A,N/A,N/A\n",
                        kernelLaunchRecord.recordIndex, kernelLaunchRecord.timeStamp,
                        name.c_str(), kernelLaunchRecord.pid,
                        kernelLaunchRecord.tid, kernelLaunchRecord.devId, kernelLaunchRecord.kernelLaunchIndex);
    if (fpRes < 0) {
        std::cout << "[msleaks] Error: Fail to write data to csv file, errno:" << fpRes << std::endl;
        return false;
    }
    return true;
}

bool DumpRecord::DumpMstxData(const ClientId &clientId, const MstxRecord &mstxRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile_, dirPath_, fileNamePrefix_, LEAKS_HEADERS)) {
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
    int fpRes = fprintf(leaksDataFile_, "%lu,%lu,mstx,%s,%lu,%lu,%d,N/A,N/A,N/A,N/A,N/A,N/A\n",
                        mstxRecord.recordIndex, mstxRecord.timeStamp, name.c_str(), mstxRecord.pid,
                        mstxRecord.tid, mstxRecord.devId);
    if (fpRes < 0) {
        std::cout << "[msleaks] Error: Fail to write data to csv file, errno:" << fpRes << std::endl;
        return false;
    }
    return true;
}

bool DumpRecord::DumpAclItfData(const ClientId &clientId, const AclItfRecord &aclItfRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile_, dirPath_, fileNamePrefix_, LEAKS_HEADERS)) {
        return false;
    }
    std::string name;
    switch (aclItfRecord.type) {
        case AclOpType::INIT:
            name = "Init";
            break;
        case AclOpType::FINALIZE:
            name = "Finalize";
            break;
        default:
            name = "N/A";
            break;
    }
    int fpRes = fprintf(leaksDataFile_, "%lu,%lu,aclItfRecord,%s,%lu,%lu,N/A,N/A,N/A,N/A,N/A,N/A,N/A\n",
                        aclItfRecord.recordIndex, aclItfRecord.timeStamp, name.c_str(), aclItfRecord.pid,
                        aclItfRecord.tid);
    if (fpRes < 0) {
        std::cout << "[msleaks] Error: Fail to write data to csv file, errno:" << fpRes << std::endl;
        return false;
    }
    return true;
}

bool DumpRecord::DumpMemPoolData(const ClientId &clientId, const EventRecord &eventRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile_, dirPath_, fileNamePrefix_, LEAKS_HEADERS)) {
        return false;
    }

    MemoryUsage memoryUsage { };
    std::string memPoolType { };
    if (eventRecord.type == RecordType::TORCH_NPU_RECORD) {
        memoryUsage = eventRecord.record.torchNpuRecord.memoryUsage;
        memPoolType = "pytorch";
    } else {
        memoryUsage = eventRecord.record.atbMemPoolRecord.memoryUsage;
        memPoolType = "atb";
    }
    auto record = eventRecord.type == RecordType::TORCH_NPU_RECORD ?
        eventRecord.record.torchNpuRecord : eventRecord.record.atbMemPoolRecord;
    std::string eventType = memoryUsage.allocSize >= 0 ? "malloc" : "free";
    int fpRes = fprintf(leaksDataFile_, "%lu,%lu,%s,%s,%lu,%lu,%d,%lu,N/A,%ld,%ld,%ld,%ld\n",
                        record.recordIndex, record.timeStamp, memPoolType.c_str(), eventType.c_str(),
                        record.pid, record.tid, record.devId, record.kernelIndex,
                        memoryUsage.ptr, memoryUsage.allocSize, memoryUsage.totalAllocated, memoryUsage.totalReserved);
    if (fpRes < 0) {
        std::cout << "[msleaks] Error: Fail to write data to csv file, errno:" << fpRes << std::endl;
        return false;
    }
    return true;
}

DumpRecord::~DumpRecord()
{
    if (leaksDataFile_ != nullptr) {
        fclose(leaksDataFile_);
        leaksDataFile_ = nullptr;
    }
}
}