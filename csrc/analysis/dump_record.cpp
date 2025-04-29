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

DumpRecord& DumpRecord::GetInstance(Config config)
{
    static DumpRecord instance(config);
    return instance;
}

DumpRecord::DumpRecord(Config config)
{
    config_ = config;
    std::string cStack = config.enableCStack ? ",Call Stack(C)" : "";
    std::string pyStack = config.enablePyStack ? ",Call Stack(Python)" : "";
    csvHeader_ = LEAKS_HEADERS + pyStack + cStack + "\n";
    SetDirPath();
}

void DumpRecord::SetDirPath()
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    dirPath_ = Utility::g_dirPath + "/" + std::string(DUMP_FILE);
}

bool DumpRecord::DumpData(const ClientId &clientId, const Record &record)
{
    std::string pyStack(record.callStackInfo.pyStack, record.callStackInfo.pyStack + record.callStackInfo.pyLen);
    std::string cStack(record.callStackInfo.cStack, record.callStackInfo.cStack + record.callStackInfo.cLen);
    CallStackString stack{cStack, pyStack};
    switch (record.eventRecord.type) {
        case RecordType::MEMORY_RECORD: {
            auto memRecord = record.eventRecord.record.memoryRecord;
            if (!DumpMemData(clientId, memRecord, stack)) {
                return false;
            }
            break;
        }
        case RecordType::KERNEL_LAUNCH_RECORD: {
            auto kernelLaunchRecord = record.eventRecord.record.kernelLaunchRecord;
            if (!DumpKernelData(clientId, kernelLaunchRecord)) {
                return false;
            }
            break;
        }
        case RecordType::ACL_ITF_RECORD: {
            auto aclItfRecord = record.eventRecord.record.aclItfRecord;
            if (!DumpAclItfData(clientId, aclItfRecord)) {
                return false;
            }
            break;
        }
        case RecordType::ATB_MEMORY_POOL_RECORD:
        case RecordType::TORCH_NPU_RECORD: {
            if (!DumpMemPoolData(clientId, record.eventRecord, stack)) {
                return false;
            }
            break;
        }
        case RecordType::MSTX_MARK_RECORD: {
            auto mstxRecord = record.eventRecord.record.mstxRecord;
            if (!DumpMstxData(clientId, mstxRecord, stack)) {
                return false;
            }
            break;
        }
        default:
            break;
    }
    return true;
}
bool DumpRecord::DumpMemData(const ClientId &clientId, const MemOpRecord &memRecord, const CallStackString &stack)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile_, dirPath_, fileNamePrefix_, csvHeader_)) {
        return false;
    }
    uint64_t currentSize;
    if (memRecord.devType == DeviceType::CPU) {
        if (memRecord.memType == MemOpType::MALLOC) {
            hostMemSizeMap_[clientId][memRecord.addr] = memRecord.memSize;
            currentSize = memRecord.memSize;
        } else if (hostMemSizeMap_[clientId].find(memRecord.addr) != hostMemSizeMap_[clientId].end()) {
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
        deviceType = memRecord.space == MemOpSpace::HOST || memRecord.devType == DeviceType::CPU ?
                "host" : std::to_string(memRecord.devId);
    }
    if (!Utility::Fprintf(leaksDataFile_, "%lu,%lu,%s,N/A,%lu,%lu,%s,%lu,%llu,%lu,%lu,N/A,N/A",
        memRecord.recordIndex, memRecord.timeStamp, memOp.c_str(), memRecord.pid, memRecord.tid, deviceType.c_str(),
        memRecord.kernelIndex, memRecord.flag, memRecord.addr, currentSize)) {
        return false;
    }
    if (config_.enablePyStack && !Utility::Fprintf(leaksDataFile_, ",%s", stack.pyStack.c_str())) {
        return false;
    }
    if (config_.enableCStack && !Utility::Fprintf(leaksDataFile_, ",%s", stack.cStack.c_str())) {
        return false;
    }
    if (!Utility::Fprintf(leaksDataFile_, "\n")) {
        return false;
    }
    return true;
}

bool DumpRecord::DumpKernelData(const ClientId &clientId, const KernelLaunchRecord &kernelLaunchRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile_, dirPath_, fileNamePrefix_, csvHeader_)) {
        return false;
    }
    std::string name;
    if (kernelLaunchRecord.kernelName[0] == '\0') {
        name = "N/A";
    } else {
        name = kernelLaunchRecord.kernelName;
    }
    if (!Utility::Fprintf(leaksDataFile_, "%lu,%lu,kernelLaunch,%s,%lu,%lu,%d,%lu,N/A,N/A,N/A,N/A,N/A\n",
        kernelLaunchRecord.recordIndex, kernelLaunchRecord.timeStamp, name.c_str(), kernelLaunchRecord.pid,
        kernelLaunchRecord.tid, kernelLaunchRecord.devId, kernelLaunchRecord.kernelLaunchIndex)) {
        return false;
    }
    return true;
}

bool DumpRecord::DumpMstxData(const ClientId &clientId, const MstxRecord &mstxRecord, const CallStackString &stack)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile_, dirPath_, fileNamePrefix_, csvHeader_)) {
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
    if (!Utility::Fprintf(leaksDataFile_, "%lu,%lu,mstx,%s,%lu,%lu,%d,N/A,N/A,N/A,N/A,N/A,N/A",
        mstxRecord.recordIndex, mstxRecord.timeStamp, name.c_str(), mstxRecord.pid, mstxRecord.tid, mstxRecord.devId)) {
        return false;
    }
    if (config_.enablePyStack && name == "Mark" && !Utility::Fprintf(leaksDataFile_, ",%s", stack.pyStack.c_str())) {
        return false;
    }
    if (!Utility::Fprintf(leaksDataFile_, "\n")) {
        return false;
    }
    return true;
}

bool DumpRecord::DumpAclItfData(const ClientId &clientId, const AclItfRecord &aclItfRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile_, dirPath_, fileNamePrefix_, csvHeader_)) {
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
    if (!Utility::Fprintf(leaksDataFile_, "%lu,%lu,aclItfRecord,%s,%lu,%lu,N/A,N/A,N/A,N/A,N/A,N/A,N/A\n",
        aclItfRecord.recordIndex, aclItfRecord.timeStamp, name.c_str(), aclItfRecord.pid, aclItfRecord.tid)) {
        return false;
    }
    return true;
}

bool DumpRecord::DumpMemPoolData(const ClientId &clientId, const EventRecord &eventRecord, const CallStackString &stack)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile_, dirPath_, fileNamePrefix_, csvHeader_)) {
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
    if (!Utility::Fprintf(leaksDataFile_, "%lu,%lu,%s,%s,%lu,%lu,%d,%lu,N/A,%ld,%ld,%ld,%ld\n",
        record.recordIndex, record.timeStamp, memPoolType.c_str(), eventType.c_str(),
        record.pid, record.tid, record.devId, record.kernelIndex,
        memoryUsage.ptr, memoryUsage.allocSize, memoryUsage.totalAllocated, memoryUsage.totalReserved)) {
        return false;
    }
    if (config_.enablePyStack && !Utility::Fprintf(leaksDataFile_, ",%s", stack.pyStack.c_str())) {
        return false;
    }
    if (config_.enableCStack && !Utility::Fprintf(leaksDataFile_, ",%s", stack.cStack.c_str())) {
        return false;
    }
    if (!Utility::Fprintf(leaksDataFile_, "\n")) {
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