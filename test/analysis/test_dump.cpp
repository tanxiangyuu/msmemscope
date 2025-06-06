// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <string>
#define private public
#include "dump_record.h"
#include "device_manager.h"
#undef private
#include "record_info.h"
#include "config_info.h"
#include "securec.h"
#include "file.h"
#include "data_handler.h"
#include "memory_state_record.h"

using namespace Leaks;

TEST(DumpRecord, dump_cpu_memory_record_expect_success)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc = MemOpRecord{};
    memRecordMalloc.devType = DeviceType::CPU;
    memRecordMalloc.tid = 10;
    memRecordMalloc.pid = 10;
    memRecordMalloc.flag = 10;
    memRecordMalloc.modid = 55;
    memRecordMalloc.devId = 10;
    memRecordMalloc.recordIndex = 102;
    memRecordMalloc.kernelIndex = 101;
    memRecordMalloc.space = MemOpSpace::DEVICE;
    memRecordMalloc.addr = 1234;
    memRecordMalloc.memSize = 128;
    memRecordMalloc.timeStamp = 789;
    memRecordMalloc.memType = MemOpType::MALLOC;
    record.eventRecord.record.memoryRecord = memRecordMalloc;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    CallStackString stack{};
    std::shared_ptr<MemoryStateRecord> memoryStateRecord = std::make_shared<MemoryStateRecord>(config);
    std::vector<MemStateInfo> meminfoList = {};
    MemStateInfo info;
    meminfoList.push_back(info);
    memoryStateRecord->ptrMemoryInfoMap_.insert({{"common", 1234}, meminfoList});
    DeviceManager::GetInstance(config).memoryStateRecordMap_[clientId] = memoryStateRecord;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));

    record.eventRecord.record.memoryRecord.memType = MemOpType::FREE;
    
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
    
    record.eventRecord.record.memoryRecord.memType = MemOpType::MALLOC;
    memRecordMalloc.space = MemOpSpace::HOST;
    
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
    record.eventRecord.record.memoryRecord.memType = MemOpType::FREE;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
}
TEST(DumpRecord, dump_memory_record_expect_success)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc = MemOpRecord{};
    
    memRecordMalloc.tid = 10;
    memRecordMalloc.pid = 10;
    memRecordMalloc.flag = 10;
    memRecordMalloc.modid = 55;
    memRecordMalloc.devId = 10;
    memRecordMalloc.recordIndex = 102;
    memRecordMalloc.kernelIndex = 101;
    memRecordMalloc.space = MemOpSpace::DEVICE;
    memRecordMalloc.addr = 1234;
    memRecordMalloc.memSize = 128;
    memRecordMalloc.timeStamp = 789;
    memRecordMalloc.memType = MemOpType::MALLOC;
    record.eventRecord.record.memoryRecord = memRecordMalloc;
    Config config;
    ClientId clientId = 0;
    config.dataFormat = 0;
    CallStackString stack{};
    std::shared_ptr<MemoryStateRecord> memoryStateRecord = std::make_shared<MemoryStateRecord>(config);
    std::vector<MemStateInfo> meminfoList = {};
    MemStateInfo info;
    meminfoList.push_back(info);
    memoryStateRecord->ptrMemoryInfoMap_.insert({{"common", 1234}, meminfoList});
    DeviceManager::GetInstance(config).memoryStateRecordMap_[clientId] = memoryStateRecord;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
    config.enableCStack = true;
    config.enablePyStack = true;
    record.eventRecord.record.memoryRecord.memType = MemOpType::FREE;
    
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
    record.eventRecord.record.memoryRecord.memType = MemOpType::MALLOC;
    memRecordMalloc.space = MemOpSpace::HOST;
    
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
    record.eventRecord.record.memoryRecord.memType = MemOpType::FREE;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
}
TEST(DumpRecord, dump_kernelLaunch_record_expect_success)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::KERNEL_LAUNCH_RECORD;

    auto kernelLaunchRecord = KernelLaunchRecord{};
    
    kernelLaunchRecord.pid = 10;
    kernelLaunchRecord.tid = 10;
    kernelLaunchRecord.kernelLaunchIndex = 101;
    kernelLaunchRecord.recordIndex = 102;
    kernelLaunchRecord.timeStamp = 123;
    record.eventRecord.record.kernelLaunchRecord = kernelLaunchRecord;
    Config config;
    ClientId clientId = 0;
    config.dataFormat = 0;
    std::string testName = "123";
    CallStackString stack{};
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
    strncpy_s(record.eventRecord.record.kernelLaunchRecord.kernelName,
                KERNELNAME_MAX_SIZE, testName.c_str(), KERNELNAME_MAX_SIZE - 1);
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
}
TEST(DumpRecord, dump_aclItf_record_expect_success)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::ACL_ITF_RECORD;
    auto aclItfRecord = AclItfRecord{};
    
    aclItfRecord.pid = 10;
    aclItfRecord.tid = 10;
    aclItfRecord.recordIndex = 101;
    aclItfRecord.aclItfRecordIndex = 102;
    aclItfRecord.timeStamp = 123;
    aclItfRecord.type = AclOpType::INIT;
    record.eventRecord.record.aclItfRecord = aclItfRecord;
    Config config;
    ClientId clientId = 0;
    config.dataFormat = 0;
    CallStackString stack{};
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));

    aclItfRecord.type = AclOpType::FINALIZE;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
}
TEST(DumpRecord, dump_torchnpu_record_expect_success)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::TORCH_NPU_RECORD;
    auto memPoolRecord = MemPoolRecord{};
    memPoolRecord.recordIndex = 101;
    
    MemoryUsage memoryUsage;
    memoryUsage.allocSize = 128;
    memoryUsage.totalActive = 128;
    memoryUsage.totalReserved = 128;
    memoryUsage.totalAllocated = 128;
    memoryUsage.ptr = 123;
    memoryUsage.streamPtr = 123;
    memoryUsage.deviceIndex = 10;
    memoryUsage.allocatorType = 0;
    memoryUsage.dataType = 0;
    memPoolRecord.memoryUsage = memoryUsage;
    record.eventRecord.record.memPoolRecord = memPoolRecord;
    Config config;
    ClientId clientId = 0;
    config.dataFormat = 0;
    CallStackString stack{};
    std::shared_ptr<MemoryStateRecord> memoryStateRecord = std::make_shared<MemoryStateRecord>(config);
    std::vector<MemStateInfo> meminfoList = {};
    MemStateInfo info;
    meminfoList.push_back(info);
    memoryStateRecord->ptrMemoryInfoMap_.insert({{"PTA", 123}, meminfoList});
    DeviceManager::GetInstance(config).memoryStateRecordMap_[clientId] = memoryStateRecord;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
    config.enableCStack = true;
    config.enablePyStack = true;
    memoryUsage.allocSize = -128;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
}
TEST(DumpRecord, dump_empty_torchnpu_record)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::TORCH_NPU_RECORD;
    auto memPoolRecord = MemPoolRecord{};

    MemoryUsage memoryUsage;
    memPoolRecord.memoryUsage = memoryUsage;
    record.eventRecord.record.memPoolRecord = memPoolRecord;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    CallStackString stack{};
    std::shared_ptr<MemoryStateRecord> memoryStateRecord = std::make_shared<MemoryStateRecord>(config);
    std::vector<MemStateInfo> meminfoList = {};
    MemStateInfo info;
    meminfoList.push_back(info);
    memoryStateRecord->ptrMemoryInfoMap_.insert({{"PTA", 123}, meminfoList});
    DeviceManager::GetInstance(config).memoryStateRecordMap_[clientId] = memoryStateRecord;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
}
TEST(DumpRecord, dump_mindsporenpu_record_expect_success)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::MINDSPORE_NPU_RECORD;
    auto memPoolRecord = MemPoolRecord{};
    memPoolRecord.recordIndex = 101;
    
    MemoryUsage memoryUsage;
    memoryUsage.allocSize = 128;
    memoryUsage.totalActive = 128;
    memoryUsage.totalReserved = 128;
    memoryUsage.totalAllocated = 128;
    memoryUsage.ptr = 123;
    memoryUsage.streamPtr = 123;
    memoryUsage.deviceIndex = 10;
    memoryUsage.allocatorType = 0;
    memoryUsage.dataType = 0;
    memPoolRecord.memoryUsage = memoryUsage;
    record.eventRecord.record.memPoolRecord = memPoolRecord;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    CallStackString stack{};
    std::shared_ptr<MemoryStateRecord> memoryStateRecord = std::make_shared<MemoryStateRecord>(config);
    std::vector<MemStateInfo> meminfoList = {};
    MemStateInfo info;
    meminfoList.push_back(info);
    memoryStateRecord->ptrMemoryInfoMap_.insert({{"MINDSPORE", 123}, meminfoList});
    DeviceManager::GetInstance(config).memoryStateRecordMap_[clientId] = memoryStateRecord;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
    config.enableCStack = true;
    config.enablePyStack = true;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
}
TEST(DumpRecord, dump_empty_mindsporenpu_record)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::MINDSPORE_NPU_RECORD;
    auto memPoolRecord = MemPoolRecord{};

    MemoryUsage memoryUsage;
    memPoolRecord.memoryUsage = memoryUsage;
    record.eventRecord.record.memPoolRecord = memPoolRecord;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    CallStackString stack{};
    std::shared_ptr<MemoryStateRecord> memoryStateRecord = std::make_shared<MemoryStateRecord>(config);
    std::vector<MemStateInfo> meminfoList = {};
    MemStateInfo info;
    meminfoList.push_back(info);
    memoryStateRecord->ptrMemoryInfoMap_.insert({{"MINDSPORE", 123}, meminfoList});
    DeviceManager::GetInstance(config).memoryStateRecordMap_[clientId] = memoryStateRecord;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
}
TEST(DumpRecord, dump_invalid_memory_record)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc = MemOpRecord{};
    
    memRecordMalloc.tid = 10;
    memRecordMalloc.pid = 10;
    memRecordMalloc.flag = 10;
    memRecordMalloc.modid = 55;
    memRecordMalloc.devId = 10;
    memRecordMalloc.recordIndex = 102;
    memRecordMalloc.kernelIndex = 101;
    memRecordMalloc.space = MemOpSpace::INVALID;
    memRecordMalloc.addr = 0x1234;
    memRecordMalloc.memSize = 128;
    memRecordMalloc.timeStamp = 789;
    memRecordMalloc.memType = MemOpType::MALLOC;
    record.eventRecord.record.memoryRecord = memRecordMalloc;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    CallStackString stack{};
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));

    record.eventRecord.record.memoryRecord.memType = MemOpType::FREE;
    record.eventRecord.record.memoryRecord.devId = GD_INVALID_NUM;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
}
TEST(DumpRecord, dump_mstx_mark_expect_success)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::MSTX_MARK_RECORD;
    auto mstxRecord = MstxRecord{};
    
    mstxRecord.markType = MarkType::MARK_A;
    mstxRecord.timeStamp = 1234;
    mstxRecord.pid = 10;
    mstxRecord.tid = 10;
    mstxRecord.devId = 1;
    mstxRecord.stepId = 10;
    mstxRecord.streamId = 1;
    strncpy_s(mstxRecord.markMessage, sizeof(mstxRecord.markMessage), "test mark",
        sizeof(mstxRecord.markMessage) - 1);
    mstxRecord.recordIndex = 1;
    record.eventRecord.record.mstxRecord = mstxRecord;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    CallStackString stack{};
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
    config.enablePyStack = true;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
}
TEST(DumpRecord, dump_aten_launch_start_expect_success)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::ATEN_OP_LAUNCH_RECORD;
    auto atenOpLaunchRecord = AtenOpLaunchRecord{};
    
    atenOpLaunchRecord.eventType = Leaks::OpEventType::ATEN_START;
    atenOpLaunchRecord.timestamp = 1234;
    atenOpLaunchRecord.pid = 10;
    atenOpLaunchRecord.tid = 10;
    atenOpLaunchRecord.devId = 1;
    strncpy_s(atenOpLaunchRecord.name, sizeof(atenOpLaunchRecord.name),
        "leaks-aten-b: {func.__module__}.{func.__name__}", sizeof(atenOpLaunchRecord.name) - 1);
    atenOpLaunchRecord.recordIndex = 1;
    record.eventRecord.record.atenOpLaunchRecord = atenOpLaunchRecord;

    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    CallStackString stack{};
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
}
TEST(DumpRecord, dump_aten_launch_end_expect_success)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::ATEN_OP_LAUNCH_RECORD;
    auto atenOpLaunchRecord = AtenOpLaunchRecord{};
    
    atenOpLaunchRecord.eventType = Leaks::OpEventType::ATEN_END;
    atenOpLaunchRecord.timestamp = 1234;
    atenOpLaunchRecord.pid = 10;
    atenOpLaunchRecord.tid = 10;
    atenOpLaunchRecord.devId = 1;
    strncpy_s(atenOpLaunchRecord.name, sizeof(atenOpLaunchRecord.name),
        "{func.__module__}.{func.__name__}", sizeof(atenOpLaunchRecord.name) - 1);
    atenOpLaunchRecord.recordIndex = 1;
    record.eventRecord.record.atenOpLaunchRecord = atenOpLaunchRecord;

    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    CallStackString stack{};
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
}
TEST(DumpRecord, dump_aten_launch_expect_success)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::MEM_ACCESS_RECORD;
    auto memAccessRecord = MemAccessRecord{};
    
    memAccessRecord.eventType = Leaks::AccessType::READ;
    memAccessRecord.timestamp = 1234;
    memAccessRecord.pid = 10;
    memAccessRecord.tid = 10;
    memAccessRecord.devId = 1;
    strncpy_s(memAccessRecord.name, sizeof(memAccessRecord.name),
        "{func.__module__}.{func.__name__}", sizeof(memAccessRecord.name) - 1);
    strncpy_s(memAccessRecord.name, sizeof(memAccessRecord.name),
        "{size:100,shape:([3.3])", sizeof(memAccessRecord.name) - 1);
    memAccessRecord.recordIndex = 1;
    record.eventRecord.record.memAccessRecord = memAccessRecord;

    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    CallStackString stack{};
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
}
TEST(DumpRecord, dump_mstx_range_start_expect_success)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::MSTX_MARK_RECORD;
    auto mstxRecord = MstxRecord{};
    
    mstxRecord.markType = MarkType::RANGE_START_A;
    mstxRecord.timeStamp = 5678;
    mstxRecord.pid = 10;
    mstxRecord.tid = 10;
    mstxRecord.devId = 2;
    mstxRecord.stepId = 2;
    mstxRecord.streamId = 123;
    strncpy_s(mstxRecord.markMessage, sizeof(mstxRecord.markMessage), "test range start",
        sizeof(mstxRecord.markMessage) - 1);
    mstxRecord.recordIndex = 1;
    record.eventRecord.record.mstxRecord = mstxRecord;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    CallStackString stack{};
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
}
TEST(DumpRecord, dump_mstx_range_end_expect_success)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::MSTX_MARK_RECORD;
    auto mstxRecord = MstxRecord{};
    
    mstxRecord.markType = MarkType::RANGE_END;
    mstxRecord.timeStamp = 7890;
    mstxRecord.pid = 10;
    mstxRecord.tid = 10;
    mstxRecord.devId = 3;
    mstxRecord.stepId = -1;
    mstxRecord.streamId = 123;
    mstxRecord.recordIndex = 1;
    record.eventRecord.record.mstxRecord = mstxRecord;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    CallStackString stack{};
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
}

TEST(DumpRecord, dump_atb_op_start_expect_success)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::ATB_OP_EXECUTE_RECORD;
    auto atbOpExecuteRecord = AtbOpExecuteRecord{};
    
    atbOpExecuteRecord.eventType = OpEventType::ATB_START;
    strncpy_s(atbOpExecuteRecord.name, sizeof(atbOpExecuteRecord.name),
              "ElewiseOperation", sizeof(atbOpExecuteRecord.name) - 1);
    strncpy_s(atbOpExecuteRecord.params, sizeof(atbOpExecuteRecord.params),
              "{path:0_784115/0/0_ElewiseOperation,workspace ptr:0,workspace size:0}",
              sizeof(atbOpExecuteRecord.params) - 1);
    atbOpExecuteRecord.timestamp = 7890;
    atbOpExecuteRecord.pid = 10;
    atbOpExecuteRecord.tid = 11;
    atbOpExecuteRecord.devId = 3;
    atbOpExecuteRecord.recordIndex = 1;
    record.eventRecord.record.atbOpExecuteRecord = atbOpExecuteRecord;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    CallStackString stack{};
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
}

TEST(DumpRecord, dump_atb_op_end_expect_success)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::ATB_OP_EXECUTE_RECORD;
    auto atbOpExecuteRecord = AtbOpExecuteRecord{};
    
    atbOpExecuteRecord.eventType = OpEventType::ATB_END;
    strncpy_s(atbOpExecuteRecord.name, sizeof(atbOpExecuteRecord.name),
              "ElewiseOperation", sizeof(atbOpExecuteRecord.name) - 1);
    strncpy_s(atbOpExecuteRecord.params, sizeof(atbOpExecuteRecord.params),
              "{path:0_784115/0/0_ElewiseOperation,workspace ptr:0,workspace size:0}",
              sizeof(atbOpExecuteRecord.params) - 1);
    atbOpExecuteRecord.timestamp = 7890;
    atbOpExecuteRecord.pid = 10;
    atbOpExecuteRecord.tid = 11;
    atbOpExecuteRecord.devId = 3;
    atbOpExecuteRecord.recordIndex = 1;
    record.eventRecord.record.atbOpExecuteRecord = atbOpExecuteRecord;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    CallStackString stack{};
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
}

TEST(DumpRecord, dump_atb_kernel_start_expect_success)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::ATB_KERNEL_RECORD;
    auto atbKernelRecord = AtbKernelRecord{};
    
    atbKernelRecord.eventType = KernelEventType::KERNEL_START;
    strncpy_s(atbKernelRecord.name, sizeof(atbKernelRecord.name),
              "0_AddI32Kernel", sizeof(atbKernelRecord.name) - 1);
    strncpy_s(atbKernelRecord.params, sizeof(atbKernelRecord.params),
              "{path:0_784115/0/0_ElewiseOperation/0_AddI32Kernel}",
              sizeof(atbKernelRecord.params) - 1);
    atbKernelRecord.timestamp = 7890;
    atbKernelRecord.pid = 10;
    atbKernelRecord.tid = 11;
    atbKernelRecord.devId = 3;
    atbKernelRecord.recordIndex = 1;
    record.eventRecord.record.atbKernelRecord = atbKernelRecord;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    CallStackString stack{};
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
}

TEST(DumpRecord, dump_atb_kernel_end_expect_success)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::ATB_KERNEL_RECORD;
    auto atbKernelRecord = AtbKernelRecord{};
    
    atbKernelRecord.eventType = KernelEventType::KERNEL_END;
    strncpy_s(atbKernelRecord.name, sizeof(atbKernelRecord.name),
              "0_AddI32Kernel", sizeof(atbKernelRecord.name) - 1);
    strncpy_s(atbKernelRecord.params, sizeof(atbKernelRecord.params),
              "{path:0_784115/0/0_ElewiseOperation/0_AddI32Kernel}",
              sizeof(atbKernelRecord.params) - 1);
    atbKernelRecord.timestamp = 7890;
    atbKernelRecord.pid = 10;
    atbKernelRecord.tid = 11;
    atbKernelRecord.devId = 3;
    atbKernelRecord.recordIndex = 1;
    record.eventRecord.record.atbKernelRecord = atbKernelRecord;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    CallStackString stack{};
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record, stack));
}

TEST(DumpRecord, dump_kernel_execute_data_expect_success)
{
    auto kernelExcuteRecord = KernelExcuteRecord{};
    Config config;
    config.dataFormat = 0;
    kernelExcuteRecord.recordIndex = 1;
    kernelExcuteRecord.devId = 0;
    kernelExcuteRecord.type = KernelEventType::KERNEL_START;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpKernelExcuteData(kernelExcuteRecord));
}

TEST(DumpRecord, set_alloc_attr_expect_success)
{
    Config config;
    MemStateInfo memInfo = {};
    memInfo.attr.addr = 1000;
    memInfo.attr.size = 1234;
    memInfo.attr.modid = 3;
    memInfo.attr.leaksDefinedOwner = "HCCL";
    memInfo.container.eventType = "HAL";
    DumpRecord::GetInstance(config).SetAllocAttr(memInfo);
    ASSERT_EQ(memInfo.container.attr, "\"{addr:1000,size:1234,MID:3,owner:HCCL}\"");

    memInfo.container.eventType = "PTA";
    memInfo.attr.totalAllocated = 100;
    memInfo.attr.totalReserved = 100;
    memInfo.attr.leaksDefinedOwner = "PTA";
    DumpRecord::GetInstance(config).SetAllocAttr(memInfo);
    ASSERT_EQ(memInfo.container.attr, "\"{addr:1000,size:1234,total:100,used:100,owner:PTA}\"");
}
