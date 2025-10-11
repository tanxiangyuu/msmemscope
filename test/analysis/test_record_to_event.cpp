// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <string>
#include <unistd.h>
#include <iostream>
#define private public
#include "process.h"
#undef private
#include "record_info.h"
#include "config_info.h"
#include "securec.h"
#include "file.h"
#include "event.h"
#include "event_dispatcher.h"
#include "memory_state_manager.h"
 
using namespace Leaks;

TEST(TestRecordToEvent, check_device_str_is_valid)
{
    std::string device1 = "N/A";
    EXPECT_TRUE(IsInvalidDevice(device1));
    std::string device2 = "";
    EXPECT_TRUE(IsInvalidDevice(device2));
    std::string device3 = "host";
    EXPECT_FALSE(IsInvalidDevice(device3));
}
 
TEST(TestRecordToEvent, transfer_hal_device_malloc_free)
{
    Config config;
    Process process(config);
 
    auto record1 = MemOpRecord{};
    record1.type = RecordType::MEMORY_RECORD;
    record1.subtype = RecordSubType::MALLOC;
    record1.space = MemOpSpace::DEVICE;
    record1.recordIndex = 1;
    record1.timestamp = 12;
    record1.pid = 123;
    record1.tid = 1234;
    record1.modid = 0;
    record1.devId = 0;
    record1.addr = 0x1234;
    record1.memSize = 128;
 
    auto event1 = process.RecordToEvent(static_cast<RecordBase*>(&record1));
    EXPECT_EQ(event1->poolType, PoolType::HAL);
    EXPECT_EQ(event1->device, "0");
 
    auto record2 = MemOpRecord{};
    record2.type = RecordType::MEMORY_RECORD;
    record2.subtype = RecordSubType::FREE;
    record2.space = MemOpSpace::DEVICE;
    record2.recordIndex = 1;
    record2.timestamp = 12;
    record2.pid = 123;
    record2.tid = 1234;
    record2.modid = 0;
    record2.devId = GD_INVALID_NUM;
    record2.addr = 0x1234;
    record2.memSize = 0;
 
    auto event2 = process.RecordToEvent(static_cast<RecordBase*>(&record2));
    EXPECT_EQ(event2->poolType, PoolType::HAL);
    EXPECT_EQ(event2->device, "N/A");
}

TEST(TestRecordToEvent, transfer_pta_caching_malloc_free)
{
    Config config;
    Process process(config);
 
    auto record1 = MemPoolRecord{};
    record1.type = RecordType::PTA_CACHING_POOL_RECORD;
    record1.recordIndex = 1;
    record1.timestamp = 12;
    record1.pid = 123;
    record1.tid = 1234;
    record1.memoryUsage.dataType = 0;
    record1.memoryUsage.ptr = 0x1234;
    record1.memoryUsage.deviceIndex = 1;
    record1.memoryUsage.allocSize = 10;
    record1.memoryUsage.totalReserved = 10;
    record1.memoryUsage.totalAllocated = 10;
 
    auto event1 = std::dynamic_pointer_cast<MemoryEvent>(
        process.RecordToEvent(static_cast<RecordBase*>(&record1)));
    EXPECT_EQ(event1->poolType, PoolType::PTA_CACHING);
    EXPECT_EQ(event1->device, "1");
    EXPECT_EQ(event1->eventType, EventBaseType::MALLOC);
    EXPECT_EQ(event1->describeOwner, "");

    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<MemPoolRecord>(
        TLVBlockType::ADDR_OWNER, "leaks_mem",
        TLVBlockType::CALL_STACK_C, "c_file: c_func()",
        TLVBlockType::CALL_STACK_PYTHON, "py_file: py_func1()\npy_file: py_func2()");
    auto& record2 = *(buffer.Cast<MemPoolRecord>());
    record2.type = RecordType::PTA_CACHING_POOL_RECORD;
    record2.recordIndex = 1;
    record2.timestamp = 12;
    record2.pid = 123;
    record2.tid = 1234;
    record2.memoryUsage.dataType = 1;
    record2.memoryUsage.ptr = 0x1234;
    record2.memoryUsage.deviceIndex = 1;
    record2.memoryUsage.allocSize = 10;
    record2.memoryUsage.totalReserved = 0;
    record2.memoryUsage.totalAllocated = 0;
 
    auto event2 = process.RecordToEvent(static_cast<RecordBase*>(&record2));
    EXPECT_EQ(event2->poolType, PoolType::PTA_CACHING);
    EXPECT_EQ(event2->device, "1");
    EXPECT_EQ(event2->eventType, EventBaseType::FREE);
}

TEST(TestRecordToEvent, transfer_pta_workspace_malloc_free)
{
    Config config;
    Process process(config);
 
    auto record1 = MemPoolRecord{};
    record1.type = RecordType::PTA_WORKSPACE_POOL_RECORD;
    record1.recordIndex = 1;
    record1.timestamp = 12;
    record1.pid = 123;
    record1.tid = 1234;
    record1.memoryUsage.dataType = 0;
    record1.memoryUsage.ptr = 0x1234;
    record1.memoryUsage.deviceIndex = 1;
    record1.memoryUsage.allocSize = 10;
    record1.memoryUsage.totalReserved = 10;
    record1.memoryUsage.totalAllocated = 10;
 
    auto event1 = std::dynamic_pointer_cast<MemoryEvent>(
        process.RecordToEvent(static_cast<RecordBase*>(&record1)));
    EXPECT_EQ(event1->poolType, PoolType::PTA_WORKSPACE);
    EXPECT_EQ(event1->device, "1");
    EXPECT_EQ(event1->eventType, EventBaseType::MALLOC);
    EXPECT_EQ(event1->describeOwner, "");
 
    auto record2 = MemPoolRecord{};
    record2.type = RecordType::PTA_WORKSPACE_POOL_RECORD;
    record2.recordIndex = 1;
    record2.timestamp = 12;
    record2.pid = 123;
    record2.tid = 1234;
    record2.memoryUsage.dataType = 1;
    record2.memoryUsage.ptr = 0x1234;
    record2.memoryUsage.deviceIndex = 1;
    record2.memoryUsage.allocSize = 10;
    record2.memoryUsage.totalReserved = 0;
    record2.memoryUsage.totalAllocated = 0;
 
    auto event2 = process.RecordToEvent(static_cast<RecordBase*>(&record2));
    EXPECT_EQ(event2->poolType, PoolType::PTA_WORKSPACE);
    EXPECT_EQ(event2->device, "1");
    EXPECT_EQ(event2->eventType, EventBaseType::FREE);
}

TEST(TestRecordToEvent, transfer_atb_malloc_free)
{
    Config config;
    Process process(config);
 
    auto record1 = MemPoolRecord{};
    record1.type = RecordType::ATB_MEMORY_POOL_RECORD;
    record1.recordIndex = 1;
    record1.timestamp = 12;
    record1.pid = 123;
    record1.tid = 1234;
    record1.memoryUsage.dataType = 0;
    record1.memoryUsage.ptr = 0x1234;
    record1.memoryUsage.deviceIndex = 1;
    record1.memoryUsage.allocSize = 10;
    record1.memoryUsage.totalReserved = 10;
    record1.memoryUsage.totalAllocated = 10;
 
    auto event1 = std::dynamic_pointer_cast<MemoryEvent>(
        process.RecordToEvent(static_cast<RecordBase*>(&record1)));
    EXPECT_EQ(event1->poolType, PoolType::ATB);
    EXPECT_EQ(event1->device, "1");
    EXPECT_EQ(event1->eventType, EventBaseType::MALLOC);
    EXPECT_EQ(event1->describeOwner, "");
 
    auto record2 = MemPoolRecord{};
    record2.type = RecordType::ATB_MEMORY_POOL_RECORD;
    record2.recordIndex = 1;
    record2.timestamp = 12;
    record2.pid = 123;
    record2.tid = 1234;
    record2.memoryUsage.dataType = 1;
    record2.memoryUsage.ptr = 0x1234;
    record2.memoryUsage.deviceIndex = 1;
    record2.memoryUsage.allocSize = 10;
    record2.memoryUsage.totalReserved = 0;
    record2.memoryUsage.totalAllocated = 0;
 
    auto event2 = process.RecordToEvent(static_cast<RecordBase*>(&record2));
    EXPECT_EQ(event2->poolType, PoolType::ATB);
    EXPECT_EQ(event2->device, "1");
    EXPECT_EQ(event2->eventType, EventBaseType::FREE);
}
 
TEST(TestRecordToEvent, transfer_mindspore_malloc_free)
{
    Config config;
    Process process(config);
 
    auto record1 = MemPoolRecord{};
    record1.type = RecordType::MINDSPORE_NPU_RECORD;
    record1.recordIndex = 1;
    record1.timestamp = 12;
    record1.pid = 123;
    record1.tid = 1234;
    record1.memoryUsage.dataType = 0;
    record1.memoryUsage.ptr = 0x1234;
    record1.memoryUsage.deviceIndex = 1;
    record1.memoryUsage.allocSize = 10;
    record1.memoryUsage.totalReserved = 10;
    record1.memoryUsage.totalAllocated = 10;
 
    auto event1 = std::dynamic_pointer_cast<MemoryEvent>(
        process.RecordToEvent(static_cast<RecordBase*>(&record1)));
    EXPECT_EQ(event1->poolType, PoolType::MINDSPORE);
    EXPECT_EQ(event1->device, "1");
    EXPECT_EQ(event1->eventType, EventBaseType::MALLOC);
    EXPECT_EQ(event1->describeOwner, "");
 
    auto record2 = MemPoolRecord{};
    record2.type = RecordType::MINDSPORE_NPU_RECORD;
    record2.recordIndex = 1;
    record2.timestamp = 12;
    record2.pid = 123;
    record2.tid = 1234;
    record2.memoryUsage.dataType = 1;
    record2.memoryUsage.ptr = 0x1234;
    record2.memoryUsage.deviceIndex = 1;
    record2.memoryUsage.allocSize = 10;
    record2.memoryUsage.totalReserved = 0;
    record2.memoryUsage.totalAllocated = 0;
 
    auto event2 = process.RecordToEvent(static_cast<RecordBase*>(&record2));
    EXPECT_EQ(event2->poolType, PoolType::MINDSPORE);
    EXPECT_EQ(event2->device, "1");
    EXPECT_EQ(event2->eventType, EventBaseType::FREE);
}
 
TEST(TestRecordToEvent, transfer_aten_access)
{
    Config config;
    Process process(config);
 
    auto record1 = MemAccessRecord{};
    record1.type = RecordType::MEM_ACCESS_RECORD;
    record1.recordIndex = 1;
    record1.timestamp = 12;
    record1.pid = 123;
    record1.tid = 1234;
    record1.memType = AccessMemType::ATEN;
    record1.eventType = AccessType::READ;
    record1.addr = 0x1234;
    record1.devId = 1;
    record1.memSize = 10;
    auto event1 = process.RecordToEvent(static_cast<RecordBase*>(&record1));
    EXPECT_EQ(event1->eventSubType, EventSubType::ATEN_READ);
    EXPECT_EQ(event1->name, "N/A");
 
    auto record2 = MemAccessRecord{};
    record2.type = RecordType::MEM_ACCESS_RECORD;
    record2.recordIndex = 1;
    record2.timestamp = 12;
    record2.pid = 123;
    record2.tid = 1234;
    record2.memType = AccessMemType::ATEN;
    record2.eventType = AccessType::WRITE;
    record2.addr = 0x1234;
    record2.devId = 1;
    record2.memSize = 10;
    auto event2 = process.RecordToEvent(static_cast<RecordBase*>(&record2));
    EXPECT_EQ(event2->eventSubType, EventSubType::ATEN_WRITE);
 
    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<MemAccessRecord>(
        TLVBlockType::OP_NAME, "aten.add",
        TLVBlockType::MEM_ATTR, "dtype:torch.float16,shape:([1,2])",
        TLVBlockType::CALL_STACK_C, "c_file: c_func()",
        TLVBlockType::CALL_STACK_PYTHON, "py_file: py_func1()\npy_file: py_func2()");
    auto& record3 = *(buffer.Cast<MemAccessRecord>());
    record3.type = RecordType::MEM_ACCESS_RECORD;
    record3.recordIndex = 1;
    record3.timestamp = 12;
    record3.pid = 123;
    record3.tid = 1234;
    record3.memType = AccessMemType::ATEN;
    record3.eventType = AccessType::UNKNOWN;
    record3.addr = 0x1234;
    record3.devId = 1;
    record3.memSize = 10;
    auto event3 = process.RecordToEvent(static_cast<RecordBase*>(&record3));
    EXPECT_EQ(event3->eventSubType, EventSubType::ATEN_READ_OR_WRITE);
}
 
TEST(TestRecordToEvent, transfer_atb_access)
{
    Config config;
    Process process(config);
 
    auto record1 = MemAccessRecord{};
    record1.type = RecordType::MEM_ACCESS_RECORD;
    record1.recordIndex = 1;
    record1.timestamp = 12;
    record1.pid = 123;
    record1.tid = 1234;
    record1.memType = AccessMemType::ATB;
    record1.eventType = AccessType::READ;
    record1.addr = 0x1234;
    record1.devId = 1;
    record1.memSize = 10;
    auto event1 = process.RecordToEvent(static_cast<RecordBase*>(&record1));
    EXPECT_EQ(event1->eventSubType, EventSubType::ATB_READ);
 
    auto record2 = MemAccessRecord{};
    record2.type = RecordType::MEM_ACCESS_RECORD;
    record2.recordIndex = 1;
    record2.timestamp = 12;
    record2.pid = 123;
    record2.tid = 1234;
    record2.memType = AccessMemType::ATB;
    record2.eventType = AccessType::WRITE;
    record2.addr = 0x1234;
    record2.devId = 1;
    record2.memSize = 10;
    auto event2 = process.RecordToEvent(static_cast<RecordBase*>(&record2));
    EXPECT_EQ(event2->eventSubType, EventSubType::ATB_WRITE);
 
    auto record3 = MemAccessRecord{};
    record3.type = RecordType::MEM_ACCESS_RECORD;
    record3.recordIndex = 1;
    record3.timestamp = 12;
    record3.pid = 123;
    record3.tid = 1234;
    record3.memType = AccessMemType::ATB;
    record3.eventType = AccessType::UNKNOWN;
    record3.addr = 0x1234;
    record3.devId = 1;
    record3.memSize = 10;
    auto event3 = process.RecordToEvent(static_cast<RecordBase*>(&record3));
    EXPECT_EQ(event3->eventSubType, EventSubType::ATB_READ_OR_WRITE);
}
 
TEST(TestRecordToEvent, transfer_owner_event)
{
    Config config;
    Process process(config);
 
    auto record1 = AddrInfo{};
    record1.type = RecordType::ADDR_INFO_RECORD;
    record1.subtype = RecordSubType::USER_DEFINED;
    record1.recordIndex = 1;
    record1.timestamp = 12;
    record1.pid = 123;
    record1.tid = 1234;
    record1.addr = 0x1234;
 
    auto event1 = std::dynamic_pointer_cast<MemoryOwnerEvent>(
        process.RecordToEvent(static_cast<RecordBase*>(&record1)));
    EXPECT_EQ(event1->eventSubType, EventSubType::DESCRIBE_OWNER);
    EXPECT_EQ(event1->owner, "");
 
    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<AddrInfo>(
        TLVBlockType::ADDR_OWNER, "model@gradient");
    auto& record2 = *(buffer.Cast<AddrInfo>());
    record2.type = RecordType::ADDR_INFO_RECORD;
    record2.subtype = RecordSubType::PTA_OPTIMIZER_STEP;
    record2.recordIndex = 1;
    record2.timestamp = 12;
    record2.pid = 123;
    record2.tid = 1234;
    record2.addr = 0x1234;
 
    auto event2 = process.RecordToEvent(static_cast<RecordBase*>(&record2));
    EXPECT_EQ(event2->eventSubType, EventSubType::TORCH_OPTIMIZER_STEP_OWNER);
}
 
TEST(TestRecordToEvent, transfer_atb_op_launch)
{
    Config config;
    Process process(config);
 
    auto record1 = AtbOpExecuteRecord{};
    record1.type = RecordType::ATB_OP_EXECUTE_RECORD;
    record1.recordIndex = 1;
    record1.timestamp = 12;
    record1.pid = 123;
    record1.tid = 1234;
    record1.subtype = RecordSubType::ATB_START;
    record1.devId = 1;
 
    auto event1 = process.RecordToEvent(static_cast<RecordBase*>(&record1));
    EXPECT_EQ(event1->eventSubType, EventSubType::ATB_START);
    EXPECT_EQ(event1->device, "1");
 
    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<AtbOpExecuteRecord>(
        TLVBlockType::ATB_NAME, "operation",
        TLVBlockType::ATB_PARAMS, "dtype:A,format:B,shape:[1,2,]");
    auto& record2 = *(buffer.Cast<AtbOpExecuteRecord>());
    record2.type = RecordType::ATB_OP_EXECUTE_RECORD;
    record2.recordIndex = 1;
    record2.timestamp = 12;
    record2.pid = 123;
    record2.tid = 1234;
    record2.subtype = RecordSubType::ATB_END;
    record2.devId = GD_INVALID_NUM;
 
    auto event2 = process.RecordToEvent(static_cast<RecordBase*>(&record2));
    EXPECT_EQ(event2->eventSubType, EventSubType::ATB_END);
    EXPECT_EQ(event2->device, "N/A");
}
 
TEST(TestRecordToEvent, transfer_aten_op_launch)
{
    Config config;
    Process process(config);
 
    auto record1 = AtenOpLaunchRecord{};
    record1.type = RecordType::ATEN_OP_LAUNCH_RECORD;
    record1.recordIndex = 1;
    record1.timestamp = 12;
    record1.pid = 123;
    record1.tid = 1234;
    record1.subtype = RecordSubType::ATEN_START;
    record1.devId = 1;
 
    auto event1 = process.RecordToEvent(static_cast<RecordBase*>(&record1));
    EXPECT_EQ(event1->eventSubType, EventSubType::ATEN_START);
    EXPECT_EQ(event1->device, "1");
 
    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<AtenOpLaunchRecord>(
        TLVBlockType::ATEN_NAME, "aten.add",
        TLVBlockType::CALL_STACK_C, "c_file: c_func()",
        TLVBlockType::CALL_STACK_PYTHON, "py_file: py_func1()\npy_file: py_func2()");
    auto& record2 = *(buffer.Cast<AtenOpLaunchRecord>());
    record2.type = RecordType::ATEN_OP_LAUNCH_RECORD;
    record2.recordIndex = 1;
    record2.timestamp = 12;
    record2.pid = 123;
    record2.tid = 1234;
    record2.subtype = RecordSubType::ATEN_END;
    record2.devId = GD_INVALID_NUM;
 
    auto event2 = process.RecordToEvent(static_cast<RecordBase*>(&record2));
    EXPECT_EQ(event2->eventSubType, EventSubType::ATEN_END);
    EXPECT_EQ(event2->device, "N/A");
}
 
TEST(TestRecordToEvent, transfer_kernel_launch)
{
    Config config;
    Process process(config);
 
    auto record1 = KernelLaunchRecord{};
    record1.type = RecordType::KERNEL_LAUNCH_RECORD;
    record1.recordIndex = 1;
    record1.timestamp = 12;
    record1.pid = 123;
    record1.tid = 1234;
    record1.devId = 1;
    record1.streamId = 1;
    record1.taskId = 2;
 
    auto event1 = std::dynamic_pointer_cast<KernelLaunchEvent>(
        process.RecordToEvent(static_cast<RecordBase*>(&record1)));
    EXPECT_EQ(event1->device, "1");
    EXPECT_EQ(event1->streamId, "1");
    EXPECT_EQ(event1->taskId, "2");
 
    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<KernelLaunchRecord>(
        TLVBlockType::KERNEL_NAME, "kernel");
    auto& record2 = *(buffer.Cast<KernelLaunchRecord>());
    record2.type = RecordType::KERNEL_LAUNCH_RECORD;
    record2.recordIndex = 1;
    record2.timestamp = 12;
    record2.pid = 123;
    record2.tid = 1234;
    record2.devId = GD_INVALID_NUM;
    record2.streamId = 1;
    record2.taskId = 2;
 
    auto event2 = process.RecordToEvent(static_cast<RecordBase*>(&record2));
    EXPECT_EQ(event2->device, "N/A");
}
 
TEST(TestRecordToEvent, transfer_kernel_execute)
{
    Config config;
    Process process(config);
 
    auto record1 = KernelExcuteRecord{};
    record1.type = RecordType::KERNEL_EXCUTE_RECORD;
    record1.recordIndex = 1;
    record1.timestamp = 12;
    record1.pid = 123;
    record1.tid = 1234;
    record1.subtype = RecordSubType::KERNEL_START;
    record1.devId = 1;
    record1.streamId = 1;
    record1.taskId = 2;
 
    auto event1 = process.RecordToEvent(static_cast<RecordBase*>(&record1));
    EXPECT_EQ(event1->device, "1");
 
    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<KernelExcuteRecord>(
        TLVBlockType::KERNEL_NAME, "kernel");
    auto& record2 = *(buffer.Cast<KernelExcuteRecord>());
    record2.type = RecordType::KERNEL_EXCUTE_RECORD;
    record2.recordIndex = 1;
    record2.timestamp = 12;
    record2.pid = 123;
    record2.tid = 1234;
    record2.subtype = RecordSubType::KERNEL_END;
    record2.devId = GD_INVALID_NUM;
    record2.streamId = 1;
    record2.taskId = 2;
 
    auto event2 = process.RecordToEvent(static_cast<RecordBase*>(&record2));
    EXPECT_EQ(event2->device, "N/A");
}
 
TEST(TestRecordToEvent, transfer_atb_kernel_execute)
{
    Config config;
    Process process(config);
 
    auto record1 = AtbKernelRecord{};
    record1.type = RecordType::ATB_KERNEL_RECORD;
    record1.recordIndex = 1;
    record1.timestamp = 12;
    record1.pid = 123;
    record1.tid = 1234;
    record1.subtype = RecordSubType::KERNEL_START;
    record1.devId = 1;
 
    auto event1 = process.RecordToEvent(static_cast<RecordBase*>(&record1));
    EXPECT_EQ(event1->device, "1");
 
    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<AtbKernelRecord>(
        TLVBlockType::ATB_NAME, "operation",
        TLVBlockType::ATB_PARAMS, "path:0_111/0/operation");
    auto& record2 = *(buffer.Cast<AtbKernelRecord>());
    record2.type = RecordType::ATB_KERNEL_RECORD;
    record2.recordIndex = 1;
    record2.timestamp = 12;
    record2.pid = 123;
    record2.tid = 1234;
    record2.subtype = RecordSubType::KERNEL_END;
    record2.devId = GD_INVALID_NUM;
 
    auto event2 = process.RecordToEvent(static_cast<RecordBase*>(&record2));
    EXPECT_EQ(event2->device, "N/A");
}
 
TEST(TestRecordToEvent, transfer_mstx_event)
{
    Config config;
    Process process(config);
 
    auto record1 = MstxRecord{};
    record1.type = RecordType::MSTX_MARK_RECORD;
    record1.recordIndex = 1;
    record1.timestamp = 12;
    record1.pid = 123;
    record1.tid = 1234;
    record1.markType = MarkType::MARK_A;
    record1.devId = 1;
    auto event1 = process.RecordToEvent(static_cast<RecordBase*>(&record1));
    EXPECT_EQ(event1->eventSubType, EventSubType::MSTX_MARK);
    EXPECT_EQ(event1->device, "1");
 
    RecordBuffer buffer1 = RecordBuffer::CreateRecordBuffer<MstxRecord>(
        TLVBlockType::MARK_MESSAGE, "operation=",
        TLVBlockType::CALL_STACK_C, "c_file: c_func()",
        TLVBlockType::CALL_STACK_PYTHON, "py_file: py_func1()\npy_file: py_func2()");
    auto record2 = *(buffer1.Cast<MstxRecord>());
    record2.type = RecordType::MSTX_MARK_RECORD;
    record2.recordIndex = 1;
    record2.timestamp = 12;
    record2.pid = 123;
    record2.tid = 1234;
    record2.markType = MarkType::RANGE_START_A;
    record2.devId = GD_INVALID_NUM;
    auto event2 = process.RecordToEvent(static_cast<RecordBase*>(&record2));
    EXPECT_EQ(event2->eventSubType, EventSubType::MSTX_RANGE_START);
    EXPECT_EQ(event2->device, "N/A");
 
    RecordBuffer buffer2 = RecordBuffer::CreateRecordBuffer<MstxRecord>(
        TLVBlockType::MARK_MESSAGE, "operation",
        TLVBlockType::CALL_STACK_C, "c_file: c_func()",
        TLVBlockType::CALL_STACK_PYTHON, "py_file: py_func1()\npy_file: py_func2()");
    auto& record3 = *(buffer2.Cast<MstxRecord>());
    record3.type = RecordType::MSTX_MARK_RECORD;
    record3.recordIndex = 1;
    record3.timestamp = 12;
    record3.pid = 123;
    record3.tid = 1234;
    record3.markType = MarkType::RANGE_END;
    record3.devId = 3;
 
    auto event3 = process.RecordToEvent(static_cast<RecordBase*>(&record3));
    EXPECT_EQ(event3->eventSubType, EventSubType::MSTX_RANGE_END);
    EXPECT_EQ(event3->device, "3");
}
 
TEST(TestRecordToEvent, transfer_system_event)
{
    Config config;
    Process process(config);
 
    auto record1 = AclItfRecord{};
    record1.type = RecordType::ACL_ITF_RECORD;
    record1.recordIndex = 1;
    record1.timestamp = 12;
    record1.pid = 123;
    record1.tid = 1234;
    record1.subtype = RecordSubType::INIT;
 
    auto event1 = process.RecordToEvent(static_cast<RecordBase*>(&record1));
    EXPECT_EQ(event1->eventSubType, EventSubType::ACL_INIT);
 
    auto record2 = AclItfRecord{};
    record2.type = RecordType::ACL_ITF_RECORD;
    record2.recordIndex = 1;
    record2.timestamp = 12;
    record2.pid = 123;
    record2.tid = 1234;
    record2.subtype = RecordSubType::FINALIZE;
    auto event2 = process.RecordToEvent(static_cast<RecordBase*>(&record2));
    EXPECT_EQ(event2->eventSubType, EventSubType::ACL_FINI);
}

TEST(TestRecordToEvent, transfer_trace_event)
{
    Config config;
    Process process(config);
 
    auto record1 = TraceStatusRecord{};
    record1.type = RecordType::TRACE_STATUS_RECORD;
    record1.recordIndex = 1;
    record1.timestamp = 12;
    record1.pid = 123;
    record1.tid = 1234;
    record1.status = static_cast<uint8_t>(EventTraceStatus::IN_TRACING);
 
    auto event1 = process.RecordToEvent(static_cast<RecordBase*>(&record1));
    EXPECT_EQ(event1->eventSubType, EventSubType::TRACE_START);
 
    auto record2 = TraceStatusRecord{};
    record2.type = RecordType::TRACE_STATUS_RECORD;
    record2.recordIndex = 1;
    record2.timestamp = 12;
    record2.pid = 123;
    record2.tid = 1234;
    record2.status = static_cast<uint8_t>(EventTraceStatus::NOT_IN_TRACING);
    auto event2 = process.RecordToEvent(static_cast<RecordBase*>(&record2));
    EXPECT_EQ(event2->eventSubType, EventSubType::TRACE_STOP);
}