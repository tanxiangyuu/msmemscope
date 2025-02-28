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

TEST(DumpRecord, dump_memory_record_expect_success)
{
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
    EXPECT_TRUE(DumpRecord::GetInstance().DumpData(clientId, record));

    record.record.memoryRecord.memType = MemOpType::FREE;
    
    EXPECT_TRUE(DumpRecord::GetInstance().DumpData(clientId, record));
    
    record.record.memoryRecord.memType = MemOpType::MALLOC;
    memRecordMalloc.space = MemOpSpace::HOST;
    
    EXPECT_TRUE(DumpRecord::GetInstance().DumpData(clientId, record));
    record.record.memoryRecord.memType = MemOpType::FREE;
    EXPECT_TRUE(DumpRecord::GetInstance().DumpData(clientId, record));
}
TEST(DumpRecord, dump_kernelLaunch_record_expect_success)
{
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
    EXPECT_TRUE(DumpRecord::GetInstance().DumpData(clientId, record));
}
TEST(DumpRecord, dump_aclItf_record_expect_success)
{
    auto record = EventRecord{};
    record.type = RecordType::ACL_ITF_RECORD;
    auto aclItfRecord = AclItfRecord{};
    
    aclItfRecord.pid = 10;
    aclItfRecord.tid = 10;
    aclItfRecord.recordIndex = 101;
    aclItfRecord.aclItfRecordIndex = 102;
    aclItfRecord.timeStamp = 123;
    record.record.aclItfRecord = aclItfRecord;
    
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance().DumpData(clientId, record));
}
TEST(DumpRecord, dump_torchnpu_record_expect_success)
{
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
    EXPECT_TRUE(DumpRecord::GetInstance().DumpData(clientId, record));
}
TEST(DumpRecord, dump_empty_torchnpu_record)
{
    auto record = EventRecord{};
    record.type = RecordType::TORCH_NPU_RECORD;
    auto torchNpuRecord = TorchNpuRecord{};

    MemoryUsage memoryUsage;
    torchNpuRecord.memoryUsage = memoryUsage;
    record.record.torchNpuRecord = torchNpuRecord;

    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance().DumpData(clientId, record));
}
TEST(DumpRecord, dump_invalid_memory_record)
{
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
    EXPECT_TRUE(DumpRecord::GetInstance().DumpData(clientId, record));

    record.record.memoryRecord.memType = MemOpType::FREE;
    EXPECT_TRUE(DumpRecord::GetInstance().DumpData(clientId, record));
}
TEST(DumpRecord, dump_msxt_mark_expect_success)
{
    auto record = EventRecord{};
    record.type = RecordType::MSTX_MARK_RECORD;
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
    
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance().DumpData(clientId, record));
}
TEST(DumpRecord, dump_msxt_range_start_expect_success)
{
    auto record = EventRecord{};
    record.type = RecordType::MSTX_MARK_RECORD;
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
    
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance().DumpData(clientId, record));
}
TEST(DumpRecord, dump_msxt_range_end_expect_success)
{
    auto record = EventRecord{};
    record.type = RecordType::MSTX_MARK_RECORD;
    auto mstxRecord = MstxRecord{};
    
    mstxRecord.markType = MarkType::RANGE_END;
    mstxRecord.timeStamp = 7890;
    mstxRecord.pid = 10;
    mstxRecord.tid = 10;
    mstxRecord.devId = 3;
    mstxRecord.stepId = -1;
    mstxRecord.streamId = 123;
    mstxRecord.recordIndex = 1;
    
    ClientId clientId = 0;
    EXPECT_TRUE(DumpRecord::GetInstance().DumpData(clientId, record));
}

TEST(DumpRecord, set_dir_path)
{
    Utility::SetDirPath("/MyPath", std::string(OUTPUT_PATH));
    DumpRecord::GetInstance().SetDirPath();
    EXPECT_EQ(DumpRecord::GetInstance().dirPath_, "/MyPath/" + std::string(DUMP_FILE));
}