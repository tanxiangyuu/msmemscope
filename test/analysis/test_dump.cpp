// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <string>
#define private public
#include "dump_record.h"
#undef private
#include "record_info.h"
#include "config_info.h"
#include "securec.h"
#include "file.h"

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
    memRecordMalloc.addr = 0x1234;
    memRecordMalloc.memSize = 128;
    memRecordMalloc.timeStamp = 789;
    memRecordMalloc.memType = MemOpType::MALLOC;
    record.eventRecord.record.memoryRecord = memRecordMalloc;
    Config config;
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record));

    record.eventRecord.record.memoryRecord.memType = MemOpType::FREE;
    
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record));
    
    record.eventRecord.record.memoryRecord.memType = MemOpType::MALLOC;
    memRecordMalloc.space = MemOpSpace::HOST;
    
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record));
    record.eventRecord.record.memoryRecord.memType = MemOpType::FREE;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record));
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
    memRecordMalloc.addr = 0x1234;
    memRecordMalloc.memSize = 128;
    memRecordMalloc.timeStamp = 789;
    memRecordMalloc.memType = MemOpType::MALLOC;
    record.eventRecord.record.memoryRecord = memRecordMalloc;
    Config config;
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record));
    config.enableCStack = true;
    config.enablePyStack = true;
    record.eventRecord.record.memoryRecord.memType = MemOpType::FREE;
    
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record));
    record.eventRecord.record.memoryRecord.memType = MemOpType::MALLOC;
    memRecordMalloc.space = MemOpSpace::HOST;
    
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record));
    record.eventRecord.record.memoryRecord.memType = MemOpType::FREE;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record));
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
    std::string testName = "123";
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record));
    strncpy_s(record.eventRecord.record.kernelLaunchRecord.kernelName,
                KERNELNAME_MAX_SIZE, testName.c_str(), KERNELNAME_MAX_SIZE - 1);
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record));
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
    record.eventRecord.record.aclItfRecord = aclItfRecord;
    Config config;
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record));
}
TEST(DumpRecord, dump_torchnpu_record_expect_success)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::TORCH_NPU_RECORD;
    auto torchNpuRecord = TorchNpuRecord{};
    torchNpuRecord.recordIndex = 101;
    
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
    torchNpuRecord.memoryUsage = memoryUsage;
    record.eventRecord.record.torchNpuRecord = torchNpuRecord;
    Config config;
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record));
    config.enableCStack = true;
    config.enablePyStack = true;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record));
}
TEST(DumpRecord, dump_empty_torchnpu_record)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::TORCH_NPU_RECORD;
    auto torchNpuRecord = TorchNpuRecord{};

    MemoryUsage memoryUsage;
    torchNpuRecord.memoryUsage = memoryUsage;
    record.eventRecord.record.torchNpuRecord = torchNpuRecord;
    Config config;
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record));
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
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record));

    record.eventRecord.record.memoryRecord.memType = MemOpType::FREE;
    record.eventRecord.record.memoryRecord.devId = GD_INVALID_NUM;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record));
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
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record));
    config.enablePyStack = true;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record));
}
TEST(DumpRecord, dump_operator_launch_start_expect_success)
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
    strncpy_s(mstxRecord.markMessage, sizeof(mstxRecord.markMessage), "func start {func.__module__}.{func.__name__}",
        sizeof(mstxRecord.markMessage) - 1);
    mstxRecord.recordIndex = 1;
    record.eventRecord.record.mstxRecord = mstxRecord;

    Config config;
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record));
}
TEST(DumpRecord, dump_operator_launch_end_expect_success)
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
    strncpy_s(mstxRecord.markMessage, sizeof(mstxRecord.markMessage), "func end {func.__module__}.{func.__name__}",
        sizeof(mstxRecord.markMessage) - 1);
    mstxRecord.recordIndex = 1;
    record.eventRecord.record.mstxRecord = mstxRecord;

    Config config;
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record));
}
TEST(DumpRecord, dump_tensor_launch_expect_success)
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
    strncpy_s(mstxRecord.markMessage, sizeof(mstxRecord.markMessage),
        "tensor:ptr={data_ptr};shape={value.shape};dtype={value.dtype};device={value.device}",
        sizeof(mstxRecord.markMessage) - 1);
    mstxRecord.recordIndex = 1;
    record.eventRecord.record.mstxRecord = mstxRecord;

    Config config;
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record));
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
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record));
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
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance(config).DumpData(clientId, record));
}

TEST(DumpRecord, set_dir_path)
{
    Config config;
    Utility::SetDirPath("/MyPath", std::string(OUTPUT_PATH));
    DumpRecord::GetInstance(config).SetDirPath();
    EXPECT_EQ(DumpRecord::GetInstance(config).dirPath_, "/MyPath/" + std::string(DUMP_FILE));
}