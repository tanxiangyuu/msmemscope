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
    auto buffer = RecordBuffer::CreateRecordBuffer<MemOpRecord>();
    MemOpRecord* record = buffer.Cast<MemOpRecord>();
    record->type = RecordType::MEMORY_RECORD;
    record->devType = DeviceType::CPU;
    record->tid = 10;
    record->pid = 10;
    record->flag = 10;
    record->modid = 55;
    record->devId = 10;
    record->recordIndex = 102;
    record->kernelIndex = 101;
    record->space = MemOpSpace::DEVICE;
    record->addr = 1234;
    record->memSize = 128;
    record->timestamp = 789;
    record->subtype = RecordSubType::MALLOC;

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
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));

    record->subtype = RecordSubType::FREE;
    
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
    
    record->subtype = RecordSubType::MALLOC;
    record->space = MemOpSpace::HOST;
    
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
    record->subtype = RecordSubType::FREE;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
}
TEST(DumpRecord, dump_memory_record_expect_success)
{
    auto buffer = RecordBuffer::CreateRecordBuffer<MemOpRecord>();
    MemOpRecord* record = buffer.Cast<MemOpRecord>();
    record->type = RecordType::MEMORY_RECORD;
    record->tid = 10;
    record->pid = 10;
    record->flag = 10;
    record->modid = 55;
    record->devId = 10;
    record->recordIndex = 102;
    record->kernelIndex = 101;
    record->space = MemOpSpace::DEVICE;
    record->addr = 1234;
    record->memSize = 128;
    record->timestamp = 789;
    record->subtype = RecordSubType::MALLOC;
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
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
    config.enableCStack = true;
    config.enablePyStack = true;
    record->subtype = RecordSubType::FREE;
    
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
    record->subtype = RecordSubType::MALLOC;
    record->space = MemOpSpace::HOST;
    
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
    record->subtype = RecordSubType::FREE;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
}
TEST(DumpRecord, dump_kernelLaunch_record_expect_success)
{
    auto buffer = RecordBuffer::CreateRecordBuffer<KernelLaunchRecord>();
    KernelLaunchRecord* record = buffer.Cast<KernelLaunchRecord>();
    record->type = RecordType::KERNEL_LAUNCH_RECORD;
    record->pid = 10;
    record->tid = 10;
    record->kernelLaunchIndex = 101;
    record->recordIndex = 102;
    record->timestamp = 123;
    Config config;
    ClientId clientId = 0;
    config.dataFormat = 0;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
    std::string testName = "123";
    auto buffer1 = RecordBuffer::CreateRecordBuffer<KernelLaunchRecord>(TLVBlockType::KERNEL_NAME, testName.c_str());
    KernelLaunchRecord* record1 = buffer1.Cast<KernelLaunchRecord>();
    record1->type = RecordType::KERNEL_LAUNCH_RECORD;
    record1->pid = 10;
    record1->tid = 10;
    record1->kernelLaunchIndex = 101;
    record1->recordIndex = 102;
    record1->timestamp = 123;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record1)));
}
TEST(DumpRecord, dump_aclItf_record_expect_success)
{
    auto buffer = RecordBuffer::CreateRecordBuffer<AclItfRecord>();
    AclItfRecord* record = buffer.Cast<AclItfRecord>();
    record->type = RecordType::ACL_ITF_RECORD;
    record->pid = 10;
    record->tid = 10;
    record->recordIndex = 101;
    record->aclItfRecordIndex = 102;
    record->timestamp = 123;
    Config config;
    ClientId clientId = 0;
    config.dataFormat = 0;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
    record->subtype = RecordSubType::FINALIZE;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
}
TEST(DumpRecord, dump_pta_caching_record_expect_success)
{
    auto buffer = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* record = buffer.Cast<MemPoolRecord>();
    record->type = RecordType::PTA_CACHING_POOL_RECORD;
    record->recordIndex = 101;
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
    record->memoryUsage = memoryUsage;
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
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
    config.enableCStack = true;
    config.enablePyStack = true;
    memoryUsage.dataType = 1;
    memoryUsage.allocSize = 128;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
}
TEST(DumpRecord, dump_empty_pta_caching_record)
{
    auto buffer = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* record = buffer.Cast<MemPoolRecord>();
    record->type = RecordType::PTA_CACHING_POOL_RECORD;
    MemoryUsage memoryUsage;
    record->memoryUsage = memoryUsage;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    std::shared_ptr<MemoryStateRecord> memoryStateRecord = std::make_shared<MemoryStateRecord>(config);
    std::vector<MemStateInfo> meminfoList = {};
    MemStateInfo info;
    meminfoList.push_back(info);
    memoryStateRecord->ptrMemoryInfoMap_.insert({{"PTA", 123}, meminfoList});
    DeviceManager::GetInstance(config).memoryStateRecordMap_[clientId] = memoryStateRecord;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
}
TEST(DumpRecord, dump_pta_workspace_record_expect_success)
{
    auto buffer = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* record = buffer.Cast<MemPoolRecord>();
    record->type = RecordType::PTA_WORKSPACE_POOL_RECORD;
    record->recordIndex = 1010;
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
    record->memoryUsage = memoryUsage;
    Config config;
    ClientId clientId = 0;
    config.dataFormat = 0;
    CallStackString stack{};
    std::shared_ptr<MemoryStateRecord> memoryStateRecord = std::make_shared<MemoryStateRecord>(config);
    std::vector<MemStateInfo> meminfoList = {};
    MemStateInfo info;
    meminfoList.push_back(info);
    memoryStateRecord->ptrMemoryInfoMap_.insert({{"PTA_WORKSPACE", 123}, meminfoList});
    DeviceManager::GetInstance(config).memoryStateRecordMap_[clientId] = memoryStateRecord;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
    config.enableCStack = true;
    config.enablePyStack = true;
    memoryUsage.dataType = 1;
    memoryUsage.allocSize = 128;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
}
TEST(DumpRecord, dump_empty_pta_workspace_record)
{
    auto buffer = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* record = buffer.Cast<MemPoolRecord>();
    record->type = RecordType::PTA_WORKSPACE_POOL_RECORD;
    MemoryUsage memoryUsage;
    record->memoryUsage = memoryUsage;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    std::shared_ptr<MemoryStateRecord> memoryStateRecord = std::make_shared<MemoryStateRecord>(config);
    std::vector<MemStateInfo> meminfoList = {};
    MemStateInfo info;
    meminfoList.push_back(info);
    memoryStateRecord->ptrMemoryInfoMap_.insert({{"PTA_WORKSPACE", 123}, meminfoList});
    DeviceManager::GetInstance(config).memoryStateRecordMap_[clientId] = memoryStateRecord;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
}
TEST(DumpRecord, dump_mindsporenpu_record_expect_success)
{
    auto buffer = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* record = buffer.Cast<MemPoolRecord>();
    record->type = RecordType::MINDSPORE_NPU_RECORD;
    record->recordIndex = 101;
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
    record->memoryUsage = memoryUsage;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    std::shared_ptr<MemoryStateRecord> memoryStateRecord = std::make_shared<MemoryStateRecord>(config);
    std::vector<MemStateInfo> meminfoList = {};
    MemStateInfo info;
    meminfoList.push_back(info);
    memoryStateRecord->ptrMemoryInfoMap_.insert({{"MINDSPORE", 123}, meminfoList});
    DeviceManager::GetInstance(config).memoryStateRecordMap_[clientId] = memoryStateRecord;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
    config.enableCStack = true;
    config.enablePyStack = true;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
}
TEST(DumpRecord, dump_empty_mindsporenpu_record)
{
    auto buffer = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* record = buffer.Cast<MemPoolRecord>();
    record->type = RecordType::MINDSPORE_NPU_RECORD;
    MemoryUsage memoryUsage;
    record->memoryUsage = memoryUsage;
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
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
}
TEST(DumpRecord, dump_invalid_memory_record)
{
    auto buffer = RecordBuffer::CreateRecordBuffer<MemOpRecord>();
    MemOpRecord* record = buffer.Cast<MemOpRecord>();
    record->type = RecordType::MEMORY_RECORD;
    record->tid = 10;
    record->pid = 10;
    record->flag = 10;
    record->modid = 55;
    record->devId = 10;
    record->recordIndex = 102;
    record->kernelIndex = 101;
    record->space = MemOpSpace::INVALID;
    record->addr = 0x1234;
    record->memSize = 128;
    record->timestamp = 789;
    record->subtype = RecordSubType::MALLOC;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
    record->subtype = RecordSubType::FREE;
    record->devId = GD_INVALID_NUM;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
}
TEST(DumpRecord, dump_mstx_mark_expect_success)
{
    auto buffer = RecordBuffer::CreateRecordBuffer<MstxRecord>(TLVBlockType::MARK_MESSAGE, "test mark");
    MstxRecord* record = buffer.Cast<MstxRecord>();
    record->type = RecordType::MSTX_MARK_RECORD;
    record->markType = MarkType::MARK_A;
    record->timestamp = 1234;
    record->pid = 10;
    record->tid = 10;
    record->devId = 1;
    record->stepId = 10;
    record->streamId = 1;
    record->recordIndex = 1;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
    config.enablePyStack = true;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
}
TEST(DumpRecord, dump_aten_launch_start_expect_success)
{
    auto buffer = RecordBuffer::CreateRecordBuffer<AtenOpLaunchRecord>(
        TLVBlockType::ATEN_NAME, "leaks-aten-b: {func.__module__}.{func.__name__}");
    AtenOpLaunchRecord* record = buffer.Cast<AtenOpLaunchRecord>();
    record->type = RecordType::ATEN_OP_LAUNCH_RECORD;
    record->subtype = Leaks::RecordSubType::ATEN_START;
    record->timestamp = 1234;
    record->pid = 10;
    record->tid = 10;
    record->devId = 1;
    record->recordIndex = 1;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
}
TEST(DumpRecord, dump_aten_launch_end_expect_success)
{
    auto buffer = RecordBuffer::CreateRecordBuffer<AtenOpLaunchRecord>(
        TLVBlockType::ATEN_NAME, "leaks-aten-b: {func.__module__}.{func.__name__}");
    AtenOpLaunchRecord* record = buffer.Cast<AtenOpLaunchRecord>();
    record->type = RecordType::ATEN_OP_LAUNCH_RECORD;
    record->subtype = Leaks::RecordSubType::ATEN_END;
    record->timestamp = 1234;
    record->pid = 10;
    record->tid = 10;
    record->devId = 1;
    record->recordIndex = 1;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    CallStackString stack{};
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
}
TEST(DumpRecord, dump_aten_launch_expect_success)
{
    auto buffer = RecordBuffer::CreateRecordBuffer<MemAccessRecord>(
        TLVBlockType::OP_NAME, "{func.__module__}.{func.__name__}", TLVBlockType::MEM_ATTR, "{size:100,shape:([3.3])");
    MemAccessRecord* record = buffer.Cast<MemAccessRecord>();
    record->type = RecordType::MEM_ACCESS_RECORD;
    record->eventType = Leaks::AccessType::READ;
    record->timestamp = 1234;
    record->pid = 10;
    record->tid = 10;
    record->devId = 1;
    record->recordIndex = 1;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
}
TEST(DumpRecord, dump_mstx_range_start_expect_success)
{
    auto buffer = RecordBuffer::CreateRecordBuffer<MstxRecord>(TLVBlockType::MARK_MESSAGE, "test range start");
    MstxRecord* record = buffer.Cast<MstxRecord>();
    record->type = RecordType::MSTX_MARK_RECORD;
    record->markType = MarkType::RANGE_START_A;
    record->timestamp = 5678;
    record->pid = 10;
    record->tid = 10;
    record->devId = 2;
    record->stepId = 2;
    record->streamId = 123;
    record->recordIndex = 1;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
}
TEST(DumpRecord, dump_mstx_range_end_expect_success)
{
    auto buffer = RecordBuffer::CreateRecordBuffer<MstxRecord>();
    MstxRecord* record = buffer.Cast<MstxRecord>();
    record->type = RecordType::MSTX_MARK_RECORD;
    record->markType = MarkType::RANGE_END;
    record->timestamp = 7890;
    record->pid = 10;
    record->tid = 10;
    record->devId = 3;
    record->stepId = -1;
    record->streamId = 123;
    record->recordIndex = 1;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    CallStackString stack{};
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
}

TEST(DumpRecord, dump_atb_op_start_expect_success)
{
    auto buffer = RecordBuffer::CreateRecordBuffer<AtbOpExecuteRecord>(
        TLVBlockType::ATB_NAME, "ElewiseOperation",
        TLVBlockType::ATB_PARAMS, "{path:0_784115/0/0_ElewiseOperation,workspace ptr:0,workspace size:0}");
    AtbOpExecuteRecord* record = buffer.Cast<AtbOpExecuteRecord>();
    record->type = RecordType::ATB_OP_EXECUTE_RECORD;
    record->subtype = RecordSubType::ATB_START;
    record->timestamp = 7890;
    record->pid = 10;
    record->tid = 11;
    record->devId = 3;
    record->recordIndex = 1;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
}

TEST(DumpRecord, dump_atb_op_end_expect_success)
{
    auto buffer = RecordBuffer::CreateRecordBuffer<AtbOpExecuteRecord>(
        TLVBlockType::ATB_NAME, "ElewiseOperation",
        TLVBlockType::ATB_PARAMS, "{path:0_784115/0/0_ElewiseOperation,workspace ptr:0,workspace size:0}");
    AtbOpExecuteRecord* record = buffer.Cast<AtbOpExecuteRecord>();
    record->type = RecordType::ATB_OP_EXECUTE_RECORD;
    record->subtype = RecordSubType::ATB_END;
    record->timestamp = 7890;
    record->pid = 10;
    record->tid = 11;
    record->devId = 3;
    record->recordIndex = 1;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
}

TEST(DumpRecord, dump_atb_kernel_start_expect_success)
{
    auto buffer = RecordBuffer::CreateRecordBuffer<AtbKernelRecord>(
        TLVBlockType::ATB_NAME, "0_AddI32Kernel",
        TLVBlockType::ATB_PARAMS, "{path:0_784115/0/0_ElewiseOperation/0_AddI32Kernel}");
    AtbKernelRecord* record = buffer.Cast<AtbKernelRecord>();
    record->type = RecordType::ATB_KERNEL_RECORD;
    record->subtype = RecordSubType::KERNEL_START;
    record->timestamp = 7890;
    record->pid = 10;
    record->tid = 11;
    record->devId = 3;
    record->recordIndex = 1;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
}

TEST(DumpRecord, dump_atb_kernel_end_expect_success)
{
    auto buffer = RecordBuffer::CreateRecordBuffer<AtbKernelRecord>(
        TLVBlockType::ATB_NAME, "0_AddI32Kernel",
        TLVBlockType::ATB_PARAMS, "{path:0_784115/0/0_ElewiseOperation/0_AddI32Kernel}");
    AtbKernelRecord* record = buffer.Cast<AtbKernelRecord>();
    record->type = RecordType::ATB_KERNEL_RECORD;
    record->subtype = RecordSubType::KERNEL_END;
    record->timestamp = 7890;
    record->pid = 10;
    record->tid = 11;
    record->devId = 3;
    record->recordIndex = 1;
    Config config;
    config.dataFormat = 0;
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, static_cast<const RecordBase*>(record)));
}

TEST(DumpRecord, dump_kernel_execute_data_expect_success)
{
    auto kernelExcuteRecord = KernelExcuteRecord{};
    Config config;
    config.dataFormat = 0;
    kernelExcuteRecord.recordIndex = 1;
    kernelExcuteRecord.devId = 0;
    kernelExcuteRecord.subtype = RecordSubType::KERNEL_START;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpKernelExcuteData(&kernelExcuteRecord));
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
