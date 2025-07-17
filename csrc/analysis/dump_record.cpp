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

bool DumpRecord::DumpData(const ClientId &clientId, const RecordBase *record)
{
    switch (record->type) {
        case RecordType::ACL_ITF_RECORD: {
            auto aclItfRecord = static_cast<const AclItfRecord*>(record);
            return DumpAclItfData(clientId, aclItfRecord);
        }
        case RecordType::MEMORY_RECORD: {
            auto memRecord = static_cast<const MemOpRecord*>(record);
            return DumpMemData(clientId, memRecord);
        }
        case RecordType::KERNEL_LAUNCH_RECORD: {
            auto kernelLaunchRecord = static_cast<const KernelLaunchRecord*>(record);
            return DumpKernelData(clientId, kernelLaunchRecord);
        }
        case RecordType::KERNEL_EXCUTE_RECORD: {
            auto kernelExcuteRecord = static_cast<const KernelExcuteRecord*>(record);
            return DumpKernelExcuteData(kernelExcuteRecord);
        }
        case RecordType::ATB_OP_EXECUTE_RECORD: {
            auto atbOpExecuteRecord = static_cast<const AtbOpExecuteRecord*>(record);
            return DumpAtbOpData(clientId, atbOpExecuteRecord);
        }
        case RecordType::ATB_MEMORY_POOL_RECORD:
        case RecordType::TORCH_NPU_RECORD:
        case RecordType::MINDSPORE_NPU_RECORD: {
            auto memPoolRecord = static_cast<const MemPoolRecord*>(record);
            return DumpMemPoolData(clientId, memPoolRecord);
        }
        case RecordType::MSTX_MARK_RECORD: {
            const MstxRecord* mstxRecord = static_cast<const MstxRecord*>(record);
            return DumpMstxData(clientId, mstxRecord);
        }
        case RecordType::ATB_KERNEL_RECORD: {
            auto atbKernelRecord = static_cast<const AtbKernelRecord*>(record);
            return DumpAtbKernelData(clientId, atbKernelRecord);
        }
        case RecordType::ATEN_OP_LAUNCH_RECORD:{
            auto atenOpLaunchRecord = static_cast<const AtenOpLaunchRecord*>(record);
            return DumpAtenOpLaunchData(clientId, atenOpLaunchRecord);
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

bool DumpRecord::DumpMemData(const ClientId &clientId, const MemOpRecord *memRecord)
{
    if (memRecord->subtype == RecordSubType::MALLOC) {
        return true;
    }
    if (!handler_->Init()) {
        return false;
    }

    // free事件，落盘记录的全部内存状态数据
    auto ptr = memRecord->addr;
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

bool DumpRecord::DumpKernelData(const ClientId &clientId, const KernelLaunchRecord *kernelLaunchRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!handler_->Init()) {
        return false;
    }
    const TLVBlock* tlv = GetTlvBlock(*kernelLaunchRecord, TLVBlockType::KERNEL_NAME);
    std::string name = tlv == nullptr ? "N/A" : tlv->data;

    DumpContainer container;
    container.id = kernelLaunchRecord->recordIndex;
    container.event = "KERNEL_LAUNCH";
    container.eventType = "KERNEL_LAUNCH";
    container.name = name;
    container.timestamp = kernelLaunchRecord->timestamp;
    container.pid = kernelLaunchRecord->pid;
    container.tid = kernelLaunchRecord->tid;
    container.deviceId = std::to_string(kernelLaunchRecord->devId);
    container.addr = "N/A";

    // 组装attr属性
    std::ostringstream oss;
    oss << "{steamId:" << kernelLaunchRecord->streamId << ",taskId:" << kernelLaunchRecord->taskId << "}";
    std::string attr = "\"" + oss.str() + "\"";
    container.attr = attr;

    CallStackString emptyStack {};
    bool isWriteSuccess = handler_->Write(&container, emptyStack);
    return isWriteSuccess;
}

bool DumpRecord::DumpKernelExcuteData(const KernelExcuteRecord *record)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!handler_->Init()) {
        return false;
    }

    const TLVBlock* tlv = GetTlvBlock(*record, TLVBlockType::KERNEL_NAME);
    std::string name = tlv == nullptr ? "N/A" : tlv->data;

    DumpContainer container;
    container.id = record->recordIndex;
    container.event = "KERNEL_LAUNCH";
    container.eventType = record->subtype == RecordSubType::KERNEL_START ? "KERNEL_EXCUTE_START"
                                                                                  : "KERNEL_EXCUTE_END";
    container.name = name;
    container.timestamp = record->timestamp;
    container.pid = INVALID_PROCESSID;
    container.tid = INVALID_THREADID;
    container.deviceId = std::to_string(record->devId);
    container.addr = "N/A";

    // 组装attr属性
    std::ostringstream oss;
    oss << "{steamId:" << record->streamId << ",taskId:" << record->taskId << "}";
    std::string attr = "\"" + oss.str() + "\"";
    container.attr = attr;

    CallStackString emptyStack {};
    bool isWriteSuccess = handler_->Write(&container, emptyStack);
    return isWriteSuccess;
}

bool DumpRecord::DumpMstxData(const ClientId &clientId, const MstxRecord *mstxRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!handler_->Init()) {
        return false;
    }

    std::string markType;
    switch (mstxRecord->markType) {
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

    const TLVBlock* msgTlv = GetTlvBlock(*mstxRecord, TLVBlockType::MARK_MESSAGE);
    std::string mstxMsgString = msgTlv == nullptr ? "N/A" : msgTlv->data;
    if (Utility::CheckStrIsStartsWithInvalidChar(mstxMsgString.c_str())) {
        Utility::ToSafeString(mstxMsgString);
        LOG_ERROR("mstx msg %s is invalid!", mstxMsgString.c_str());
        mstxMsgString = "";
    }

    DumpContainer container;
    container.id = mstxRecord->recordIndex;
    container.event = "MSTX";
    container.eventType = markType;
    container.name = "\"" + mstxMsgString + "\""; // 用引号包住防止逗号影响判断
    container.timestamp = mstxRecord->timestamp;
    container.pid = mstxRecord->pid;
    container.tid = mstxRecord->tid;
    container.deviceId = std::to_string(mstxRecord->devId);
    container.addr = "N/A";

    const TLVBlock* cStackTlv = GetTlvBlock(*mstxRecord, TLVBlockType::CALL_STACK_C);
    std::string cStack = cStackTlv == nullptr ? "N/A" : cStackTlv->data;
    const TLVBlock* pyStackTlv = GetTlvBlock(*mstxRecord, TLVBlockType::CALL_STACK_PYTHON);
    std::string pyStack = pyStackTlv == nullptr ? "N/A" : pyStackTlv->data;
    CallStackString stack{cStack, pyStack};
    bool isWriteSuccess = handler_->Write(&container, stack);
    return isWriteSuccess;
}

bool DumpRecord::DumpAtenOpLaunchData(const ClientId &clientId, const AtenOpLaunchRecord *atenOpLaunchRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!handler_->Init()) {
        return false;
    }
    std::string eventType;
    switch (atenOpLaunchRecord->subtype) {
        case Leaks::RecordSubType::ATEN_START: {
            eventType = "ATEN_START";
            break;
        }
        case Leaks::RecordSubType::ATEN_END: {
            eventType = "ATEN_END";
            break;
        }
        default: {
            eventType = "N/A";
            break;
        }
    }
    const TLVBlock* nameTlv = GetTlvBlock(*atenOpLaunchRecord, TLVBlockType::ATEN_NAME);
    std::string name = nameTlv == nullptr ? "N/A" : nameTlv->data;

    DumpContainer container;
    container.id = atenOpLaunchRecord->recordIndex;
    container.event = "OP_LAUNCH";
    container.eventType = eventType;
    container.name = name;
    container.timestamp = atenOpLaunchRecord->timestamp;
    container.pid = atenOpLaunchRecord->pid;
    container.tid = atenOpLaunchRecord->tid;
    container.deviceId = std::to_string(atenOpLaunchRecord->devId);
    container.addr = "N/A";

    const TLVBlock* cStackTlv = GetTlvBlock(*atenOpLaunchRecord, TLVBlockType::CALL_STACK_C);
    std::string cStack = cStackTlv == nullptr ? "N/A" : cStackTlv->data;
    const TLVBlock* pyStackTlv = GetTlvBlock(*atenOpLaunchRecord, TLVBlockType::CALL_STACK_PYTHON);
    std::string pyStack = pyStackTlv == nullptr ? "N/A" : pyStackTlv->data;
    CallStackString stack{cStack, pyStack};
    bool isWriteSuccess = handler_->Write(&container, stack);
    return isWriteSuccess;
}

bool DumpRecord::DumpAclItfData(const ClientId &clientId, const AclItfRecord *aclItfRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!handler_->Init()) {
        return false;
    }
    std::string aclType;
    switch (aclItfRecord->subtype) {
        case RecordSubType::INIT:
            aclType = "ACL_INIT";
            break;
        case RecordSubType::FINALIZE:
            aclType = "ACL_FINI";
            break;
        default:
            aclType = "N/A";
            break;
    }

    DumpContainer container;
    container.id = aclItfRecord->recordIndex;
    container.event = "SYSTEM";
    container.eventType = aclType;
    container.name = "N/A";
    container.timestamp = aclItfRecord->timestamp;
    container.pid = aclItfRecord->pid;
    container.tid = aclItfRecord->tid;
    container.deviceId = "N/A";
    container.addr = "N/A";

    CallStackString emptyStack {};
    bool isWriteSuccess = handler_->Write(&container, emptyStack);
    return isWriteSuccess;
}

bool DumpRecord::DumpMemPoolData(const ClientId &clientId, const MemPoolRecord *memPoolRecord)
{
    // 内存事件类型为malloc
    if (memPoolRecord->memoryUsage.dataType == 0) {
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
    auto ptr = memPoolRecord->memoryUsage.ptr;
    auto key = std::make_pair(getMemPoolName(memPoolRecord->type), ptr);
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

bool DumpRecord::DumpAtbOpData(const ClientId &clientId, const AtbOpExecuteRecord *atbOpExecuteRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!handler_->Init()) {
        return false;
    }
    std::string eventType;
    switch (atbOpExecuteRecord->subtype) {
        case Leaks::RecordSubType::ATB_START: {
            eventType = "ATB_START";
            break;
        }
        case Leaks::RecordSubType::ATB_END: {
            eventType = "ATB_END";
            break;
        }
        default: {
            eventType = "N/A";
            break;
        }
    }
    const TLVBlock* nameTlv = GetTlvBlock(*atbOpExecuteRecord, TLVBlockType::ATB_NAME);
    std::string name = nameTlv == nullptr ? "N/A" : nameTlv->data;
    const TLVBlock* paramsTlv = GetTlvBlock(*atbOpExecuteRecord, TLVBlockType::ATB_PARAMS);
    std::string params = paramsTlv == nullptr ? "N/A" : paramsTlv->data;

    std::ostringstream oss;
    oss << "\"{" << params << "}\"";
    std::string attr = oss.str();

    DumpContainer container;
    container.id = atbOpExecuteRecord->recordIndex;
    container.event = "OP_LAUNCH";
    container.eventType = eventType;
    container.name = name;
    container.timestamp = atbOpExecuteRecord->timestamp;
    container.pid = atbOpExecuteRecord->pid;
    container.tid = atbOpExecuteRecord->tid;
    container.deviceId = std::to_string(atbOpExecuteRecord->devId);
    container.addr = "N/A";
    container.attr = attr;

    CallStackString emptyStack {};
    return handler_->Write(&container, emptyStack);
}

bool DumpRecord::DumpAtbKernelData(const ClientId &clientId, const AtbKernelRecord *atbKernelRecord)
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (!handler_->Init()) {
        return false;
    }
    std::string eventType;
    switch (atbKernelRecord->subtype) {
        case Leaks::RecordSubType::KERNEL_START: {
            eventType = "KERNEL_START";
            break;
        }
        case Leaks::RecordSubType::KERNEL_END: {
            eventType = "KERNEL_END";
            break;
        }
        default: {
            eventType = "N/A";
            break;
        }
    }
    const TLVBlock* nameTlv = GetTlvBlock(*atbKernelRecord, TLVBlockType::ATB_NAME);
    std::string name = nameTlv == nullptr ? "N/A" : nameTlv->data;
    const TLVBlock* paramsTlv = GetTlvBlock(*atbKernelRecord, TLVBlockType::ATB_PARAMS);
    std::string params = paramsTlv == nullptr ? "N/A" : paramsTlv->data;

    std::ostringstream oss;
    oss << "\"{" << params << "}\"";
    std::string attr = oss.str();

    DumpContainer container;
    container.id = atbKernelRecord->recordIndex;
    container.event = "KERNEL_LAUNCH";
    container.eventType = eventType;
    container.name = name;
    container.timestamp = atbKernelRecord->timestamp;
    container.pid = atbKernelRecord->pid;
    container.tid = atbKernelRecord->tid;
    container.deviceId = std::to_string(atbKernelRecord->devId);
    container.addr = "N/A";
    container.attr = attr;

    CallStackString emptyStack {};
    return handler_->Write(&container, emptyStack);
}
}