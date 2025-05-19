// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "dump_record.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include "log.h"
#include "file.h"
#include "utils.h"
#include "config_info.h"
#include "event_report.h"
#include "bit_field.h"

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
            return DumpMemData(clientId, memRecord, stack);
        }
        case RecordType::KERNEL_LAUNCH_RECORD: {
            auto kernelLaunchRecord = record.eventRecord.record.kernelLaunchRecord;
            return DumpKernelData(clientId, kernelLaunchRecord);
        }
        case RecordType::ACL_ITF_RECORD: {
            auto aclItfRecord = record.eventRecord.record.aclItfRecord;
            return DumpAclItfData(clientId, aclItfRecord);
        }
        case RecordType::ATB_MEMORY_POOL_RECORD:
        case RecordType::TORCH_NPU_RECORD: {
            return DumpMemPoolData(clientId, record.eventRecord, stack);
        }
        case RecordType::MINDSPORE_NPU_RECORD: {
            return DumpMemPoolData(clientId, record.eventRecord, stack);
        }
        case RecordType::MSTX_MARK_RECORD: {
            auto mstxRecord = record.eventRecord.record.mstxRecord;
            return DumpMstxData(clientId, mstxRecord, stack);
        }
        case RecordType::ATB_OP_EXECUTE_RECORD: {
            auto atbOpExecuteRecord = record.eventRecord.record.atbOpExecuteRecord;
            return DumpAtbOpData(clientId, atbOpExecuteRecord);
        }
        case RecordType::ATB_KERNEL_RECORD: {
            auto atbKernelRecord = record.eventRecord.record.atbKernelRecord;
            return DumpAtbKernelData(clientId, atbKernelRecord);
        }
        case RecordType::ATEN_OP_LAUNCH_RECORD:{
            auto atenOpLaunchRecord = record.eventRecord.record.atenOpLaunchRecord;
            return DumpAtenOpLaunchData(clientId, atenOpLaunchRecord, stack);
        }
        case RecordType::MEM_ACCESS_RECORD: {
            auto memAccessRecord = record.eventRecord.record.memAccessRecord;
            return DumpMemAccessData(clientId, memAccessRecord, stack);
        }
        default:
            break;
    }
    return true;
}

bool DumpRecord::WriteToFile(const DumpContainer &container, const CallStackString &stack)
{
    if (!Utility::Fprintf(leaksDataFile_, "%lu,%s,%s,%s,%lu,%lu,%lu,%s,%s,%s",
        container.id, container.event.c_str(), container.eventType.c_str(), container.name.c_str(),
        container.timeStamp, container.pid, container.tid, container.deviceId.c_str(),
        container.addr.c_str(), container.attr.c_str())) {
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
    std::string memOp = memRecord.memType == MemOpType::MALLOC ? "MALLOC" : "FREE";
    std::string deviceType;
    if (memRecord.devId == GD_INVALID_NUM) {
        deviceType = "N/A";
    } else {
        deviceType = memRecord.space == MemOpSpace::HOST || memRecord.devType == DeviceType::CPU ?
                "host" : std::to_string(memRecord.devId);
    }

    // 组装attr属性
    std::ostringstream oss;
    oss << "{addr:" << memRecord.addr << ",size:" << currentSize << ",owner:" << ",MID:" << memRecord.modid << "}";
    std::string attr = "\"" + oss.str() + "\"";

    DumpContainer container;
    container.id = memRecord.recordIndex;
    container.event = memOp;
    container.eventType = "HAL";
    container.name = "N/A";
    container.timeStamp = memRecord.timeStamp;
    container.pid = memRecord.pid;
    container.tid = memRecord.tid;
    container.deviceId = deviceType;
    container.addr = std::to_string(memRecord.addr);
    container.attr = attr;

    bool isWriteSuccess = WriteToFile(container, stack);
    return isWriteSuccess;
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

    DumpContainer container;
    container.id = kernelLaunchRecord.recordIndex;
    container.event = "KERNEL_LAUNCH";
    container.eventType = "KERNEL_LAUNCH";
    container.name = name;
    container.timeStamp = kernelLaunchRecord.timeStamp;
    container.pid = kernelLaunchRecord.pid;
    container.tid = kernelLaunchRecord.tid;
    container.deviceId = std::to_string(kernelLaunchRecord.devId);
    container.addr = "N/A";

    CallStackString emptyStack {};
    bool isWriteSuccess = WriteToFile(container, emptyStack);
    return isWriteSuccess;
}

bool DumpRecord::DumpMstxData(const ClientId &clientId, const MstxRecord &mstxRecord, const CallStackString &stack)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile_, dirPath_, fileNamePrefix_, csvHeader_)) {
        return false;
    }

    std::string markType;
    switch (mstxRecord.markType) {
        case Leaks::MarkType::MARK_A: {
            markType = "Mark";
            break;
        }
        case Leaks::MarkType::RANGE_START_A: {
            markType = "Range_start";
            break;
        }
        case Leaks::MarkType::RANGE_END: {
            markType = "Range_end";
            break;
        }
        default: {
            markType = "N/A";
            break;
        }
    }

    std::string mstxMsgString = mstxRecord.markMessage;

    DumpContainer container;
    container.id = mstxRecord.recordIndex;
    container.event = "MSTX";
    container.eventType = markType;
    container.name = "\"" + mstxMsgString + "\""; // 用引号包住防止逗号影响判断
    container.timeStamp = mstxRecord.timeStamp;
    container.pid = mstxRecord.pid;
    container.tid = mstxRecord.tid;
    container.deviceId = std::to_string(mstxRecord.devId);
    container.addr = "N/A";

    bool isWriteSuccess = WriteToFile(container, stack);
    return isWriteSuccess;
}

bool DumpRecord::DumpAtenOpLaunchData(const ClientId &clientId, const AtenOpLaunchRecord &atenOpLaunchRecord,
    const CallStackString &stack)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile_, dirPath_, fileNamePrefix_, csvHeader_)) {
        return false;
    }
    std::string eventType;
    switch (atenOpLaunchRecord.eventType) {
        case Leaks::OpEventType::ATEN_START: {
            eventType = "ATEN_START";
            break;
        }
        case Leaks::OpEventType::ATEN_END: {
            eventType = "ATEN_END";
            break;
        }
        default: {
            eventType = "N/A";
            break;
        }
    }

    DumpContainer container;
    container.id = atenOpLaunchRecord.recordIndex;
    container.event = "OP_LAUNCH";
    container.eventType = eventType;
    container.name = atenOpLaunchRecord.name;
    container.timeStamp = atenOpLaunchRecord.timestamp;
    container.pid = atenOpLaunchRecord.pid;
    container.tid = atenOpLaunchRecord.tid;
    container.deviceId = std::to_string(atenOpLaunchRecord.devId);
    container.addr = "N/A";

    return WriteToFile(container, stack);
}

bool DumpRecord::DumpAclItfData(const ClientId &clientId, const AclItfRecord &aclItfRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile_, dirPath_, fileNamePrefix_, csvHeader_)) {
        return false;
    }
    std::string aclType;
    switch (aclItfRecord.type) {
        case AclOpType::INIT:
            aclType = "ACL_INIT";
            break;
        case AclOpType::FINALIZE:
            aclType = "ACL_FINI";
            break;
        default:
            aclType = "N/A";
            break;
    }

    DumpContainer container;
    container.id = aclItfRecord.recordIndex;
    container.event = "SYSTEM";
    container.eventType = aclType;
    container.name = "N/A";
    container.timeStamp = aclItfRecord.timeStamp;
    container.pid = aclItfRecord.pid;
    container.tid = aclItfRecord.tid;
    container.deviceId = "N/A";
    container.addr = "N/A";

    CallStackString emptyStack {};
    bool isWriteSuccess = WriteToFile(container, emptyStack);
    return isWriteSuccess;
}

bool DumpRecord::DumpMemPoolData(const ClientId &clientId, const EventRecord &eventRecord, const CallStackString &stack)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile_, dirPath_, fileNamePrefix_, csvHeader_)) {
        return false;
    }

    MemoryUsage memoryUsage { };
    std::string memPoolType { };
    DumpContainer container;
    if (eventRecord.type == RecordType::TORCH_NPU_RECORD) {
        memoryUsage = eventRecord.record.torchNpuRecord.memoryUsage;
        memPoolType = "PTA";
        CopyMemPoolRecordMember(eventRecord.record.torchNpuRecord, container);
    } else if (eventRecord.type == RecordType::MINDSPORE_NPU_RECORD) {
        memoryUsage = eventRecord.record.mindsporeNpuRecord.memoryUsage;
        memPoolType = "Mindspore";
        CopyMemPoolRecordMember(eventRecord.record.mindsporeNpuRecord, container);
    } else {
        memoryUsage = eventRecord.record.atbMemPoolRecord.memoryUsage;
        memPoolType = "ATB";
        CopyMemPoolRecordMember(eventRecord.record.atbMemPoolRecord, container);
    }

    std::string eventType = memoryUsage.allocSize >= 0 ? "MALLOC" : "FREE";
    container.event = eventType;
    container.eventType = memPoolType;
    container.name = "N/A";
    container.addr = std::to_string(memoryUsage.ptr);
    // 组装attr属性
    std::ostringstream oss;
    oss << "{addr:" << memoryUsage.ptr << ",size:" << memoryUsage.allocSize << ",owner:" << ",total:" <<
        memoryUsage.totalReserved << ",used:" << memoryUsage.totalAllocated << "}";
    std::string attr = "\"" + oss.str() + "\"";
    container.attr = attr;

    bool isWriteSuccess = WriteToFile(container, stack);
    return isWriteSuccess;
}

bool DumpRecord::DumpAtbOpData(const ClientId &clientId, const AtbOpExecuteRecord &atbOpExecuteRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile_, dirPath_, fileNamePrefix_, csvHeader_)) {
        return false;
    }
    std::string eventType;
    switch (atbOpExecuteRecord.eventType) {
        case Leaks::OpEventType::ATB_START: {
            eventType = "ATB_START";
            break;
        }
        case Leaks::OpEventType::ATB_END: {
            eventType = "ATB_END";
            break;
        }
        default: {
            eventType = "N/A";
            break;
        }
    }

    std::ostringstream oss;
    oss << "\"{" << atbOpExecuteRecord.params << "}\"";
    std::string attr = oss.str();

    DumpContainer container;
    container.id = atbOpExecuteRecord.recordIndex;
    container.event = "OP_LAUNCH";
    container.eventType = eventType;
    container.name = atbOpExecuteRecord.name;
    container.timeStamp = atbOpExecuteRecord.timestamp;
    container.pid = atbOpExecuteRecord.pid;
    container.tid = atbOpExecuteRecord.tid;
    container.deviceId = std::to_string(atbOpExecuteRecord.devId);
    container.addr = "N/A";
    container.attr = attr;

    CallStackString emptyStack {};
    return WriteToFile(container, emptyStack);
}

bool DumpRecord::DumpAtbKernelData(const ClientId &clientId, const AtbKernelRecord &atbKernelRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile_, dirPath_, fileNamePrefix_, csvHeader_)) {
        return false;
    }
    std::string eventType;
    switch (atbKernelRecord.eventType) {
        case Leaks::KernelEventType::KERNEL_START: {
            eventType = "KERNEL_START";
            break;
        }
        case Leaks::KernelEventType::KERNEL_END: {
            eventType = "KERNEL_END";
            break;
        }
        default: {
            eventType = "N/A";
            break;
        }
    }

    std::ostringstream oss;
    oss << "\"{" << atbKernelRecord.params << "}\"";
    std::string attr = oss.str();

    DumpContainer container;
    container.id = atbKernelRecord.recordIndex;
    container.event = "KERNEL_LAUNCH";
    container.eventType = eventType;
    container.name = atbKernelRecord.name;
    container.timeStamp = atbKernelRecord.timestamp;
    container.pid = atbKernelRecord.pid;
    container.tid = atbKernelRecord.tid;
    container.deviceId = std::to_string(atbKernelRecord.devId);
    container.addr = "N/A";
    container.attr = attr;

    CallStackString emptyStack {};
    return WriteToFile(container, emptyStack);
}

bool DumpRecord::DumpMemAccessData(const ClientId &clientId, const MemAccessRecord &memAccessRecord,
    const CallStackString &stack)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!Utility::CreateCsvFile(&leaksDataFile_, dirPath_, fileNamePrefix_, csvHeader_)) {
        return false;
    }
    std::string eventType;
    switch (memAccessRecord.eventType) {
        case Leaks::AccessType::READ: {
            eventType = "READ";
            break;
        }
        case Leaks::AccessType::WRITE: {
            eventType = "WRITE";
            break;
        }
        default: {
            eventType = "UNKNOWN";
            break;
        }
    }

    std::ostringstream oss;
    oss << "\"{addr:" << memAccessRecord.addr << ",size:"
        << memAccessRecord.memSize << "," << memAccessRecord.attr << "}\"";
    std::string attr = oss.str();

    DumpContainer container;
    container.id = memAccessRecord.recordIndex;
    container.event = "ACCESS";
    container.eventType = eventType;
    container.name = memAccessRecord.name;
    container.timeStamp = memAccessRecord.timestamp;
    container.pid = memAccessRecord.pid;
    container.tid = memAccessRecord.tid;
    container.deviceId = std::to_string(memAccessRecord.devId);
    container.addr = std::to_string(memAccessRecord.addr);
    container.attr = attr;

    return WriteToFile(container, stack);
}

DumpRecord::~DumpRecord()
{
    if (leaksDataFile_ != nullptr) {
        fclose(leaksDataFile_);
        leaksDataFile_ = nullptr;
    }
}
}