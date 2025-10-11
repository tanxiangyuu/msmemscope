// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 
#ifndef EVENT_H
#define EVENT_H
 
#include <cstdint>
#include <string>
#include <tuple>
 
#include "data.h"
#include "state_manager.h"
#include "framework/record_info.h"
#include "utility/ustring.h"
#include "utils.h"
#include "log.h"
#include "trace_manager/event_trace_manager.h"
 
namespace Leaks {
 
enum class EventBaseType : uint8_t {
    MALLOC = 0,
    ACCESS,
    FREE,
    MEMORY_OWNER,
    MSTX,
    OP_LAUNCH,
    KERNEL_LAUNCH,
    SYSTEM,
    CLEAN_UP,
    INVALID,
};
 
enum class EventSubType : uint8_t {
    PTA_CACHING = 0,
    PTA_WORKSPACE,
    ATB,
    MINDSPORE,
    HAL,
    HOST,
 
    ATB_READ,
    ATB_WRITE,
    ATB_READ_OR_WRITE,
 
    ATEN_READ,
    ATEN_WRITE,
    ATEN_READ_OR_WRITE,
 
    DESCRIBE_OWNER,
    TORCH_OPTIMIZER_STEP_OWNER,
 
    ATB_START,
    ATB_END,
    ATEN_START,
    ATEN_END,
 
    KERNEL_LAUNCH,
    KERNEL_EXECUTE_START,
    KERNEL_EXECUTE_END,
    ATB_KERNEL_START,
    ATB_KERNEL_END,
 
    ACL_INIT,
    ACL_FINI,

    TRACE_START,
    TRACE_STOP,

    MSTX_MARK,
    MSTX_RANGE_START,
    MSTX_RANGE_END,
 
    CLEAN_UP,
 
    INVALID,
};
 
inline bool IsInvalidDevice(const std::string& device)
{
    if (device == "N/A" || device == "") {
        return true;
    }
    return false;
}
 
class EventBase : public DataBase {
public:
    Leaks::PoolType poolType = PoolType::INVALID;
    EventBaseType eventType = EventBaseType::INVALID;
    EventSubType eventSubType = EventSubType::INVALID;
    uint64_t id = 0;
    uint64_t timestamp = 0;
    uint64_t pid = 0;
    uint64_t tid = 0;
    uint64_t addr = 0;
    std::string name;
    std::string device;
    std::string attr;
    std::string cCallStack;
    std::string pyCallStack;
 
    EventBase() : DataBase(DataType::LEAKS_EVENT) {}
};
 
class MemoryEvent : public EventBase {
public:
    int64_t size = 0;
    int64_t total = 0;
    int64_t used = 0;
    uint64_t eventIndex = 0;
    int32_t moduleId = -1;
    std::string describeOwner;
 
    MemoryEvent() {}
 
    explicit MemoryEvent(MemOpRecord& record)
    {
        id = record.recordIndex;
        timestamp = record.timestamp;
        pid = record.pid;
        tid = record.tid;
        const TLVBlock* cStackBlock = GetTlvBlock(record, TLVBlockType::CALL_STACK_C);
        const TLVBlock* pyStackBlock = GetTlvBlock(record, TLVBlockType::CALL_STACK_PYTHON);
        cCallStack = cStackBlock == nullptr ? "" : std::string(cStackBlock->data);
        pyCallStack = pyStackBlock == nullptr ? "" : std::string(pyStackBlock->data);
 
        poolType = PoolType::HAL;
        eventType = record.subtype == RecordSubType::MALLOC ? EventBaseType::MALLOC : EventBaseType::FREE;
        eventSubType = EventSubType::HAL;
        addr = record.addr;
        name = "N/A";
        device = (record.devId == GD_INVALID_NUM) ? "N/A" : std::to_string(record.devId);
        // hal的free事件没有size信息
        size = record.subtype == RecordSubType::MALLOC ? static_cast<int64_t>(record.memSize) : 0;
        moduleId = record.modid;
        const TLVBlock* ownerBlock = GetTlvBlock(record, TLVBlockType::MEM_OWNER);
        describeOwner = ownerBlock == nullptr ? "" : std::string(ownerBlock->data);
    }
 
    explicit MemoryEvent(MemPoolRecord& record)
    {
        id = record.recordIndex;
        timestamp = record.timestamp;
        pid = record.pid;
        tid = record.tid;
        const TLVBlock* cStackBlock = GetTlvBlock(record, TLVBlockType::CALL_STACK_C);
        const TLVBlock* pyStackBlock = GetTlvBlock(record, TLVBlockType::CALL_STACK_PYTHON);
        cCallStack = cStackBlock == nullptr ? "" : std::string(cStackBlock->data);
        pyCallStack = pyStackBlock == nullptr ? "" : std::string(pyStackBlock->data);
 
        if (record.type == RecordType::PTA_CACHING_POOL_RECORD) {
            poolType = PoolType::PTA_CACHING;
            eventSubType = EventSubType::PTA_CACHING;
        } else if (record.type == RecordType::PTA_WORKSPACE_POOL_RECORD) {
            poolType = PoolType::PTA_WORKSPACE;
            eventSubType = EventSubType::PTA_WORKSPACE;
        } else if (record.type == RecordType::ATB_MEMORY_POOL_RECORD) {
            poolType = PoolType::ATB;
            eventSubType = EventSubType::ATB;
        } else {
            poolType = PoolType::MINDSPORE;
            eventSubType = EventSubType::MINDSPORE;
        }
        eventType = record.memoryUsage.dataType == 0 ? EventBaseType::MALLOC : EventBaseType::FREE;
        addr = record.memoryUsage.ptr;
        name = "N/A";
        device = std::to_string(record.memoryUsage.deviceIndex);
        size = record.memoryUsage.allocSize;
        total = record.memoryUsage.totalReserved;
        used = record.memoryUsage.totalAllocated;
        const TLVBlock* ownerBlock = GetTlvBlock(record, TLVBlockType::ADDR_OWNER);
        describeOwner = ownerBlock == nullptr ? "" : std::string(ownerBlock->data);
    }
 
    explicit MemoryEvent(MemAccessRecord& record)
    {
        id = record.recordIndex;
        timestamp = record.timestamp;
        pid = record.pid;
        tid = record.tid;
        const TLVBlock* cStackBlock = GetTlvBlock(record, TLVBlockType::CALL_STACK_C);
        const TLVBlock* pyStackBlock = GetTlvBlock(record, TLVBlockType::CALL_STACK_PYTHON);
        cCallStack = cStackBlock == nullptr ? "" : std::string(cStackBlock->data);
        pyCallStack = pyStackBlock == nullptr ? "" : std::string(pyStackBlock->data);
        poolType = record.memType == AccessMemType::ATEN ? PoolType::PTA_CACHING : PoolType::ATB;
        if (record.memType == AccessMemType::ATEN) {
            eventSubType = record.eventType == AccessType::READ ? EventSubType::ATEN_READ
                : record.eventType == AccessType::WRITE ? EventSubType::ATEN_WRITE
                : EventSubType::ATEN_READ_OR_WRITE;
        } else {
            eventSubType = record.eventType == AccessType::READ ? EventSubType::ATB_READ
                : record.eventType == AccessType::WRITE ? EventSubType::ATB_WRITE
                : EventSubType::ATB_READ_OR_WRITE;
        }
        eventType = EventBaseType::ACCESS;
        addr = record.addr;
        device = (record.devId == GD_INVALID_NUM) ? "N/A" : std::to_string(record.devId);
        const TLVBlock* opName = GetTlvBlock(record, TLVBlockType::OP_NAME);
        name = opName == nullptr ? "N/A" : std::string(opName->data);
        size = static_cast<int64_t>(record.memSize);
        const TLVBlock* memAttrBlock = GetTlvBlock(record, TLVBlockType::MEM_ATTR);
        attr = memAttrBlock == nullptr ? "" : std::string(memAttrBlock->data);
    }
};
 
class MemoryOwnerEvent : public EventBase {
public:
    std::string owner;
 
    MemoryOwnerEvent() {}
 
    explicit MemoryOwnerEvent(AddrInfo& record)
    {
        id = record.recordIndex;
        timestamp = record.timestamp;
        pid = record.pid;
        tid = record.tid;
        poolType = PoolType::PTA_CACHING;
        eventSubType = record.subtype == RecordSubType::USER_DEFINED
            ? EventSubType::DESCRIBE_OWNER : EventSubType::TORCH_OPTIMIZER_STEP_OWNER;
        eventType = EventBaseType::MEMORY_OWNER;
        addr = record.addr;
        device = "N/A";
        name = "N/A";
        const TLVBlock* ownerBlock = GetTlvBlock(record, TLVBlockType::ADDR_OWNER);
        owner = ownerBlock == nullptr ? "" : std::string(ownerBlock->data);
    }
};
 
class OpLaunchEvent : public EventBase {
public:
    OpLaunchEvent() {}
 
    explicit OpLaunchEvent(AtbOpExecuteRecord& record)
    {
        eventType = EventBaseType::OP_LAUNCH;
        id = record.recordIndex;
        timestamp = record.timestamp;
        pid = record.pid;
        tid = record.tid;
        eventSubType = record.subtype == RecordSubType::ATB_START
            ? EventSubType::ATB_START : EventSubType::ATB_END;
        device = (record.devId == GD_INVALID_NUM) ? "N/A" : std::to_string(record.devId);
        const TLVBlock* atbNameBlock = GetTlvBlock(record, TLVBlockType::ATB_NAME);
        name = atbNameBlock == nullptr ? "N/A" : std::string(atbNameBlock->data);
        const TLVBlock* atbParamsBlock = GetTlvBlock(record, TLVBlockType::ATB_PARAMS);
        attr = atbParamsBlock == nullptr ? "" : std::string(atbParamsBlock->data);
    }
 
    explicit OpLaunchEvent(AtenOpLaunchRecord& record)
    {
        eventType = EventBaseType::OP_LAUNCH;
        id = record.recordIndex;
        timestamp = record.timestamp;
        pid = record.pid;
        tid = record.tid;
        const TLVBlock* cStackBlock = GetTlvBlock(record, TLVBlockType::CALL_STACK_C);
        const TLVBlock* pyStackBlock = GetTlvBlock(record, TLVBlockType::CALL_STACK_PYTHON);
        cCallStack = cStackBlock == nullptr ? "" : std::string(cStackBlock->data);
        pyCallStack = pyStackBlock == nullptr ? "" : std::string(pyStackBlock->data);
        eventSubType = record.subtype == RecordSubType::ATEN_START
            ? EventSubType::ATEN_START : EventSubType::ATEN_END;
        device = (record.devId == GD_INVALID_NUM) ? "N/A" : std::to_string(record.devId);
        const TLVBlock* atenNameBlock = GetTlvBlock(record, TLVBlockType::ATEN_NAME);
        name = atenNameBlock == nullptr ? "N/A" : std::string(atenNameBlock->data);
    }
};
 
class KernelLaunchEvent : public EventBase {
public:
    std::string streamId;
    std::string taskId;
 
    KernelLaunchEvent() {}
 
    explicit KernelLaunchEvent(KernelLaunchRecord& record)
    {
        eventType = EventBaseType::KERNEL_LAUNCH;
        id = record.recordIndex;
        timestamp = record.timestamp;
        pid = record.pid;
        tid = record.tid;
        device = (record.devId == GD_INVALID_NUM) ? "N/A" : std::to_string(record.devId);
        eventSubType = EventSubType::KERNEL_LAUNCH;
        streamId = std::to_string(record.streamId);
        taskId = std::to_string(record.taskId);
        const TLVBlock* nameBlock = GetTlvBlock(record, TLVBlockType::KERNEL_NAME);
        name = nameBlock == nullptr ? "N/A" : std::string(nameBlock->data);
    }
 
    explicit KernelLaunchEvent(KernelExcuteRecord& record)
    {
        eventType = EventBaseType::KERNEL_LAUNCH;
        id = record.recordIndex;
        timestamp = record.timestamp;
        pid = INVALID_PROCESSID;
        tid = INVALID_THREADID;
        device = (record.devId == GD_INVALID_NUM) ? "N/A" : std::to_string(record.devId);
        eventSubType = record.subtype == RecordSubType::KERNEL_START
            ? EventSubType::KERNEL_EXECUTE_START : EventSubType::KERNEL_EXECUTE_END;
        streamId = std::to_string(record.streamId);
        taskId = std::to_string(record.taskId);
        const TLVBlock* nameBlock = GetTlvBlock(record, TLVBlockType::KERNEL_NAME);
        name = nameBlock == nullptr ? "N/A" : std::string(nameBlock->data);
    }
 
    explicit KernelLaunchEvent(AtbKernelRecord& record)
    {
        eventType = EventBaseType::KERNEL_LAUNCH;
        id = record.recordIndex;
        timestamp = record.timestamp;
        pid = record.pid;
        tid = record.tid;
        device = (record.devId == GD_INVALID_NUM) ? "N/A" : std::to_string(record.devId);
        eventSubType = record.subtype == RecordSubType::KERNEL_START
            ? EventSubType::ATB_KERNEL_START : EventSubType::ATB_KERNEL_END;
        const TLVBlock* nameBlock = GetTlvBlock(record, TLVBlockType::ATB_NAME);
        name = nameBlock == nullptr ? "N/A" : std::string(nameBlock->data);
        const TLVBlock* atbParamsBlock = GetTlvBlock(record, TLVBlockType::ATB_PARAMS);
        attr = atbParamsBlock == nullptr ? "" : std::string(atbParamsBlock->data);
    }
};
 
class MstxEvent : public EventBase {
public:
    uint64_t rangeId = 0;
    int32_t streamId = -1;
    uint64_t stepId = 0;
 
    MstxEvent() {}
 
    explicit MstxEvent(MstxRecord& record)
    {
        eventType = EventBaseType::MSTX;
        eventSubType = (record.markType == MarkType::MARK_A) ? EventSubType::MSTX_MARK :
            (record.markType == MarkType::RANGE_START_A) ? EventSubType::MSTX_RANGE_START :
            EventSubType::MSTX_RANGE_END;
        id = record.recordIndex;
        timestamp = record.timestamp;
        pid = record.pid;
        tid = record.tid;
        device = (record.devId == GD_INVALID_NUM) ? "N/A" : std::to_string(record.devId);
        const TLVBlock* cStackBlock = GetTlvBlock(record, TLVBlockType::CALL_STACK_C);
        const TLVBlock* pyStackBlock = GetTlvBlock(record, TLVBlockType::CALL_STACK_PYTHON);
        cCallStack = cStackBlock == nullptr ? "" : std::string(cStackBlock->data);
        pyCallStack = pyStackBlock == nullptr ? "" : std::string(pyStackBlock->data);
 
        const TLVBlock* msgBlock = GetTlvBlock(record, TLVBlockType::MARK_MESSAGE);
        std::string mstxMsgString = msgBlock == nullptr ? "N/A" : std::string(msgBlock->data);
        if (Utility::CheckStrIsStartsWithInvalidChar(mstxMsgString.c_str())) {
            Utility::ToSafeString(mstxMsgString);
            LOG_ERROR("mstx msg %s is invalid!", mstxMsgString.c_str());
            mstxMsgString = "";
        }
        name = "\"" + mstxMsgString + "\"";
    }
};
 
class SystemEvent : public EventBase {
public:
    SystemEvent() {}
 
    explicit SystemEvent(AclItfRecord& record)
    {
        id = record.recordIndex;
        timestamp = record.timestamp;
        pid = record.pid;
        tid = record.tid;
        device = "N/A";
        name = "N/A";
        eventType = EventBaseType::SYSTEM;
        eventSubType = record.subtype == RecordSubType::INIT ? EventSubType::ACL_INIT : EventSubType::ACL_FINI;
    }

    explicit SystemEvent(TraceStatusRecord& record)
    {
        id = record.recordIndex;
        timestamp = record.timestamp;
        pid = record.pid;
        tid = record.tid;
        device = "N/A";
        name = "N/A";
        eventType = EventBaseType::SYSTEM;

        eventSubType = (record.status == static_cast<uint8_t>(EventTraceStatus::IN_TRACING)) ?
            EventSubType::TRACE_START : EventSubType::TRACE_STOP;
    }
};
 
class CleanUpEvent : public EventBase {
public:
    CleanUpEvent() {}
 
    CleanUpEvent(PoolType type, uint64_t pidKey, uint64_t addrKey)
    {
        eventType = EventBaseType::CLEAN_UP;
        eventSubType = EventSubType::CLEAN_UP;
        poolType = type;
        pid = pidKey;
        addr = addrKey;
    }
};
 
}
 
#endif