// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <string>
#include "dump_record.h"
#include "record_info.h"
#include "config_info.h"

using namespace Leaks;

TEST(DumpRecord, dump_memory_record_expect_success)
{
    DumpRecord dump_;
    auto record = EventRecord{};
    record.type = RecordType::MEMORY_RECORD;
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
    record.record.memoryRecord = memRecordMalloc;
    
    ClientId clientId = 0;
    EXPECT_TRUE(dump_.DumpData(clientId, record));

    record.record.memoryRecord.memType = MemOpType::FREE;
    
    EXPECT_TRUE(dump_.DumpData(clientId, record));
    
    record.record.memoryRecord.memType = MemOpType::MALLOC;
    memRecordMalloc.space = MemOpSpace::HOST;
    
    EXPECT_TRUE(dump_.DumpData(clientId, record));
    record.record.memoryRecord.memType = MemOpType::FREE;
    EXPECT_TRUE(dump_.DumpData(clientId, record));
}
TEST(DumpRecord, dump_kernelLaunch_record_expect_success)
{
    DumpRecord dump_;
    auto record = EventRecord{};
    record.type = RecordType::KERNEL_LAUNCH_RECORD;
    auto kernelLaunchRecord = KernelLaunchRecord{};
    
    kernelLaunchRecord.pid = 10;
    kernelLaunchRecord.tid = 10;
    kernelLaunchRecord.kernelLaunchIndex = 101;
    kernelLaunchRecord.recordIndex = 102;
    kernelLaunchRecord.timeStamp = 123;
    record.record.kernelLaunchRecord = kernelLaunchRecord;
    
    ClientId clientId = 0;
    EXPECT_TRUE(dump_.DumpData(clientId, record));
}
TEST(DumpRecord, dump_aclItf_record_expect_success)
{
    DumpRecord dump_;
    auto record = EventRecord{};
    record.type = RecordType::ACL_ITF_RECORD;
    auto aclItfRecord = AclItfRecord{};
    
    aclItfRecord.pid = 10;
    aclItfRecord.tid = 10;
    aclItfRecord.recordIndex = 101;
    aclItfRecord.aclItfRecord = 102;
    aclItfRecord.timeStamp = 123;
    record.record.aclItfRecord = aclItfRecord;
    
    ClientId clientId = 0;
    EXPECT_TRUE(dump_.DumpData(clientId, record));
}
TEST(DumpRecord, dump_torchnpu_record_expect_success)
{
    DumpRecord dump_;
    auto record = EventRecord{};
    record.type = RecordType::TORCH_NPU_RECORD;
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
    record.record.torchNpuRecord = torchNpuRecord;
    
    ClientId clientId = 0;
    EXPECT_TRUE(dump_.DumpData(clientId, record));
}
TEST(DumpRecord, dump_empty_torchnpu_record)
{
    DumpRecord dump_;
    auto record = EventRecord{};
    record.type = RecordType::TORCH_NPU_RECORD;
    auto torchNpuRecord = TorchNpuRecord{};

    MemoryUsage memoryUsage;
    torchNpuRecord.memoryUsage = memoryUsage;
    record.record.torchNpuRecord = torchNpuRecord;

    ClientId clientId = 0;
    EXPECT_TRUE(dump_.DumpData(clientId, record));
}
TEST(DumpRecord, dump_invalid_memory_record)
{
    DumpRecord dump_;
    auto record = EventRecord{};
    record.type = RecordType::MEMORY_RECORD;
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
    record.record.memoryRecord = memRecordMalloc;

    ClientId clientId = 0;
    EXPECT_TRUE(dump_.DumpData(clientId, record));

    record.record.memoryRecord.memType = MemOpType::FREE;
    EXPECT_TRUE(dump_.DumpData(clientId, record));
}