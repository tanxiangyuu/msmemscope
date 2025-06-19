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
#include "data_handler.h"

#include <iostream>

namespace Leaks {
constexpr uint8_t OWNER_POS = 2;
DumpRecord& DumpRecord::GetInstance(Config config)
{
    static DumpRecord instance(config);
    return instance;
}

DumpRecord::DumpRecord(Config config)
{
    config_ = config;
    handler_ = MakeDataHandler(config_, DumpClass::LEAKS_RECORD);
}

bool DumpRecord::DumpData(const ClientId &clientId, const Record &record, const CallStackString &stack)
{
    switch (record.eventRecord.type) {
        case RecordType::MEMORY_RECORD: {
            auto memRecord = record.eventRecord.record.memoryRecord;
            return DumpMemData(clientId, memRecord);
        }
        case RecordType::KERNEL_LAUNCH_RECORD: {
            auto kernelLaunchRecord = record.eventRecord.record.kernelLaunchRecord;
            return DumpKernelData(clientId, kernelLaunchRecord);
        }
        case RecordType::KERNEL_EXCUTE_RECORD: {
            auto kernelExcuteRecord = record.eventRecord.record.kernelExcuteRecord;
            return DumpKernelExcuteData(kernelExcuteRecord);
        }
        case RecordType::ACL_ITF_RECORD: {
            auto aclItfRecord = record.eventRecord.record.aclItfRecord;
            return DumpAclItfData(clientId, aclItfRecord);
        }
        case RecordType::ATB_MEMORY_POOL_RECORD:
        case RecordType::TORCH_NPU_RECORD:
        case RecordType::MINDSPORE_NPU_RECORD: {
            return DumpMemPoolData(clientId, record.eventRecord);
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
        default:
            break;
    }
    return true;
}

bool DumpRecord::WriteToFile(DumpContainer &container, const CallStackString &stack)
{
    if (!handler_->Init()) {
        return false;
    }
    return handler_->Write(&container, stack);
}

void DumpRecord::SetAllocAttr(MemStateInfo& memInfo)
{
    std::ostringstream oss;
    oss << "{addr:" << memInfo.attr.addr << ",size:" << memInfo.attr.size;
    if (memInfo.container.eventType == "HAL") {
        oss << ",MID:" << memInfo.attr.modid;
    } else if (memInfo.container.eventType == "PTA" || memInfo.container.eventType == "MINDSPORE"
        || memInfo.container.eventType == "ATB") {
        oss << ",total:" << memInfo.attr.totalReserved << ",used:" << memInfo.attr.totalAllocated;
    }

    if (!memInfo.attr.leaksDefinedOwner.empty() || !memInfo.attr.userDefinedOwner.empty()) {
        oss << ",owner:" << memInfo.attr.leaksDefinedOwner << memInfo.attr.userDefinedOwner;
    }
    oss << "}";
    std::string attr = "\"" + oss.str() + "\"";
    memInfo.container.attr = attr;
}

bool DumpRecord::DumpMemData(const ClientId &clientId, const MemOpRecord &memRecord)
{
    if (memRecord.memType == MemOpType::MALLOC) {
        return true;
    }
    if (!handler_->Init()) {
        return false;
    }

    // free事件，落盘记录的全部内存状态数据
    auto ptr = memRecord.addr;
    auto key = std::make_pair("common", ptr);
    auto memoryStateRecord = DeviceManager::GetInstance(config_).GetMemoryStateRecord(clientId);
    auto memInfoLists = memoryStateRecord->GetPtrMemInfoList(key);
    {
        std::lock_guard<std::mutex> lock(fileMutex_);
        if (!handler_->Init()) {
            return false;
        }
        for (auto& memInfo : memInfoLists) {
            if (memInfo.container.event == "MALLOC") {
                SetAllocAttr(memInfo);
            }
            if (!handler_->Write(&memInfo.container, memInfo.stack)) return false;
        }
    }
    memoryStateRecord->DeleteMemStateInfo(key);

    return true;
}

bool DumpRecord::DumpKernelData(const ClientId &clientId, const KernelLaunchRecord &kernelLaunchRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!handler_->Init()) {
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

    // 组装attr属性
    std::ostringstream oss;
    oss << "{steamId:" << kernelLaunchRecord.streamId << ",taskId:" << kernelLaunchRecord.taskId << "}";
    std::string attr = "\"" + oss.str() + "\"";
    container.attr = attr;

    CallStackString emptyStack {};
    bool isWriteSuccess = handler_->Write(&container, emptyStack);
    return isWriteSuccess;
}

bool DumpRecord::DumpKernelExcuteData(const KernelExcuteRecord &record)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!handler_->Init()) {
        return false;
    }

    DumpContainer container;
    container.id = record.recordIndex;
    container.event = "KERNEL_LAUNCH";
    container.eventType = record.type == KernelEventType::KERNEL_START ? "KERNEL_EXCUTE_START" : "KERNEL_EXCUTE_END";
    container.name = record.kernelName;
    container.timeStamp = record.timeStamp;
    container.pid = INVALID_PROCESSID;
    container.tid = INVALID_THREADID;
    container.deviceId = std::to_string(record.devId);
    container.addr = "N/A";

    // 组装attr属性
    std::ostringstream oss;
    oss << "{steamId:" << record.streamId << ",taskId:" << record.taskId << "}";
    std::string attr = "\"" + oss.str() + "\"";
    container.attr = attr;

    CallStackString emptyStack {};
    bool isWriteSuccess = handler_->Write(&container, emptyStack);
    return isWriteSuccess;
}

bool DumpRecord::DumpMstxData(const ClientId &clientId, const MstxRecord &mstxRecord, const CallStackString &stack)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!handler_->Init()) {
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
    if (Utility::CheckStrIsStartsWithInvalidChar(mstxRecord.markMessage)) {
        mstxMsgString = "";
        LOG_ERROR("mstx msg %s is invalid!", mstxRecord.markMessage);
    }
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

    bool isWriteSuccess = handler_->Write(&container, stack);
    return isWriteSuccess;
}

bool DumpRecord::DumpAtenOpLaunchData(const ClientId &clientId, const AtenOpLaunchRecord &atenOpLaunchRecord,
    const CallStackString &stack)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!handler_->Init()) {
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
    
    bool isWriteSuccess = handler_->Write(&container, stack);
    return isWriteSuccess;
}

bool DumpRecord::DumpAclItfData(const ClientId &clientId, const AclItfRecord &aclItfRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!handler_->Init()) {
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
    bool isWriteSuccess = handler_->Write(&container, emptyStack);
    return isWriteSuccess;
}

bool DumpRecord::DumpMemPoolData(const ClientId &clientId, const EventRecord &eventRecord)
{
    // 内存事件类型为malloc
    if (eventRecord.record.memPoolRecord.memoryUsage.dataType == 0) {
        return true;
    }
    static auto getMemPoolName = [](RecordType type) -> std::string {
        if (type == RecordType::TORCH_NPU_RECORD) {
            return "PTA";
        } else if (type == RecordType::MINDSPORE_NPU_RECORD) {
            return "MINDSPORE";
        } else {
            return "ATB";
        }
    };

    // free事件，落盘记录的全部内存状态数据
    auto ptr = eventRecord.record.memPoolRecord.memoryUsage.ptr;
    auto key = std::make_pair(getMemPoolName(eventRecord.type), ptr);
    auto memoryStateRecord = DeviceManager::GetInstance(config_).GetMemoryStateRecord(clientId);
    auto memInfoLists = memoryStateRecord->GetPtrMemInfoList(key);
    {
        std::lock_guard<std::mutex> lock(fileMutex_);
        if (!handler_->Init()) {
            return false;
        }
        for (auto memInfo : memInfoLists) {
            if (memInfo.container.event == "MALLOC") {
                SetAllocAttr(memInfo);
            }
            if (!handler_->Write(&memInfo.container, memInfo.stack)) return false;
        }
    }
    memoryStateRecord->DeleteMemStateInfo(key);

    return true;
}

bool DumpRecord::DumpAtbOpData(const ClientId &clientId, const AtbOpExecuteRecord &atbOpExecuteRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!handler_->Init()) {
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
    return handler_->Write(&container, emptyStack);
}

bool DumpRecord::DumpAtbKernelData(const ClientId &clientId, const AtbKernelRecord &atbKernelRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!handler_->Init()) {
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
    return handler_->Write(&container, emptyStack);
}
}