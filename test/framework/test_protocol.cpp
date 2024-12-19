// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "framework/protocol.h"
#include "framework/serializer.h"

using namespace Leaks;

TEST(ProtocolTest, test_protocol_parse_memrecord)
{
    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};
    auto memRecord = MemOpRecord {};
    memRecord.recordIndex = 123;
    memRecord.kernelIndex = 123;
    memRecord.flag = 123;
    memRecord.pid = 345;
    memRecord.tid = 345;
    memRecord.devid = 9;
    memRecord.memType = MemOpType::MALLOC;
    memRecord.space = MemOpSpace::HOST;
    memRecord.modid = 234;
    memRecord.addr = 0x7958;
    memRecord.memSize = 1024;
    memRecord.timeStamp = 1234567;
    record.type = RecordType::MEMORY_RECORD;
    record.record.memoryRecord = memRecord;
    std::string str = Serialize(head, record);

    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody();
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.memoryRecord.recordIndex, memRecord.recordIndex);
    ASSERT_EQ(body.record.memoryRecord.addr, memRecord.addr);
    ASSERT_EQ(body.record.memoryRecord.memSize, memRecord.memSize);
    ASSERT_EQ(body.record.memoryRecord.timeStamp, memRecord.timeStamp);
    ASSERT_EQ(body.record.memoryRecord.kernelIndex, memRecord.kernelIndex);
    ASSERT_EQ(body.record.memoryRecord.flag, memRecord.flag);
    ASSERT_EQ(body.record.memoryRecord.pid, memRecord.pid);
    ASSERT_EQ(body.record.memoryRecord.tid, memRecord.tid);
    ASSERT_EQ(body.record.memoryRecord.devid, memRecord.devid);
    ASSERT_EQ(body.record.memoryRecord.memType, memRecord.memType);
    ASSERT_EQ(body.record.memoryRecord.space, memRecord.space);
    ASSERT_EQ(body.record.memoryRecord.modid, memRecord.modid);
}

TEST(ProtocolTest, test_protocol_parse_Invalidrecord)
{
    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};
    std::string str = Serialize(head, record);

    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_FALSE(result.GetPacketHead().type == PacketType::INVALID);
}

TEST(ProtocolTest, test_protocol_parse_step_stop_record)
{
    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};
    auto stepRecord = StepRecord {};
    stepRecord.recordIndex = 123;
    stepRecord.type = StepType::STOP;
    record.type = RecordType::STEP_RECORD;
    record.record.stepRecord = stepRecord;
    std::string str = Serialize(head, record);

    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody();
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.stepRecord.recordIndex, stepRecord.recordIndex);
    ASSERT_EQ(body.record.stepRecord.type, stepRecord.type);
}

TEST(ProtocolTest, test_protocol_parse_step_start_record)
{
    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};
    auto stepRecord = StepRecord {};
    stepRecord.recordIndex = 123;
    stepRecord.type = StepType::START;
    record.type = RecordType::STEP_RECORD;
    record.record.stepRecord = stepRecord;
    std::string str = Serialize(head, record);

    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody();
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.stepRecord.recordIndex, stepRecord.recordIndex);
    ASSERT_EQ(body.record.stepRecord.type, stepRecord.type);
}

TEST(ProtocolTest, test_protocol_parse_acl_itf_Init_record)
{
    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};
    auto aclItfRecord = AclItfRecord {};
    aclItfRecord.recordIndex = 13456;
    aclItfRecord.pid = 123;
    aclItfRecord.tid = 123;
    aclItfRecord.aclItfRecord = 123;
    aclItfRecord.timeStamp = 234;
    aclItfRecord.type = AclOpType::INIT;
    record.type = RecordType::ACL_ITF_RECORD;
    record.record.aclItfRecord = aclItfRecord;
    std::string str = Serialize(head, record);

    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody();
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.aclItfRecord.recordIndex, aclItfRecord.recordIndex);
    ASSERT_EQ(body.record.aclItfRecord.type, aclItfRecord.type);
    ASSERT_EQ(body.record.aclItfRecord.pid, aclItfRecord.pid);
    ASSERT_EQ(body.record.aclItfRecord.tid, aclItfRecord.tid);
    ASSERT_EQ(body.record.aclItfRecord.aclItfRecord, aclItfRecord.aclItfRecord);
    ASSERT_EQ(body.record.aclItfRecord.timeStamp, aclItfRecord.timeStamp);
}

TEST(ProtocolTest, test_protocol_parse_acl_itf_finalize_record)
{
    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};
    auto aclItfRecord = AclItfRecord {};
    aclItfRecord.recordIndex = 13456;
    aclItfRecord.pid = 123;
    aclItfRecord.tid = 123;
    aclItfRecord.aclItfRecord = 123;
    aclItfRecord.timeStamp = 234;
    aclItfRecord.type = AclOpType::FINALIZE;
    record.type = RecordType::ACL_ITF_RECORD;
    record.record.aclItfRecord = aclItfRecord;
    std::string str = Serialize(head, record);

    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody();
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.aclItfRecord.recordIndex, aclItfRecord.recordIndex);
    ASSERT_EQ(body.record.aclItfRecord.type, aclItfRecord.type);
    ASSERT_EQ(body.record.aclItfRecord.pid, aclItfRecord.pid);
    ASSERT_EQ(body.record.aclItfRecord.tid, aclItfRecord.tid);
    ASSERT_EQ(body.record.aclItfRecord.aclItfRecord, aclItfRecord.aclItfRecord);
    ASSERT_EQ(body.record.aclItfRecord.timeStamp, aclItfRecord.timeStamp);
}

TEST(ProtocolTest, test_protocol_parse_torch_npu_record)
{
    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};
    
    auto torchNpuRecord = TorchNpuRecord{};
    torchNpuRecord.memoryUsage = MemoryUsage{};
    torchNpuRecord.memoryUsage.device_type = 1;
    torchNpuRecord.memoryUsage.device_index = 2;
    torchNpuRecord.memoryUsage.data_type = 3;
    torchNpuRecord.memoryUsage.allocator_type = 4;
    torchNpuRecord.memoryUsage.ptr = 5;
    torchNpuRecord.memoryUsage.alloc_size = 6;
    torchNpuRecord.memoryUsage.total_allocated = 7;
    torchNpuRecord.memoryUsage.total_reserved = 8;
    torchNpuRecord.memoryUsage.total_active = 9;
    torchNpuRecord.memoryUsage.stream_ptr = 10;

    record.record.torchNpuRecord = torchNpuRecord;
    std::string str = Serialize(head, record);

    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody();
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.device_type, torchNpuRecord.memoryUsage.device_type);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.device_index, torchNpuRecord.memoryUsage.device_index);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.data_type, torchNpuRecord.memoryUsage.data_type);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.allocator_type, torchNpuRecord.memoryUsage.allocator_type);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.ptr, torchNpuRecord.memoryUsage.ptr);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.alloc_size, torchNpuRecord.memoryUsage.alloc_size);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.total_allocated, torchNpuRecord.memoryUsage.total_allocated);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.total_reserved, torchNpuRecord.memoryUsage.total_reserved);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.total_active, torchNpuRecord.memoryUsage.total_active);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.stream_ptr, torchNpuRecord.memoryUsage.stream_ptr);
}

TEST(ProtocolTest, test_protocol_parse_torch_npu_record_max)
{
    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};
    
    auto torchNpuRecord = TorchNpuRecord{};
    torchNpuRecord.memoryUsage = MemoryUsage{};
    torchNpuRecord.memoryUsage.device_type = 127;
    torchNpuRecord.memoryUsage.device_index = 127;
    torchNpuRecord.memoryUsage.data_type = 255;
    torchNpuRecord.memoryUsage.allocator_type = 255;
    torchNpuRecord.memoryUsage.ptr = 9223372036854775807;
    torchNpuRecord.memoryUsage.alloc_size = 9223372036854775807;
    torchNpuRecord.memoryUsage.total_allocated = 9223372036854775807;
    torchNpuRecord.memoryUsage.total_reserved = 9223372036854775807;
    torchNpuRecord.memoryUsage.total_active = 9223372036854775807;
    torchNpuRecord.memoryUsage.stream_ptr = 9223372036854775807;

    record.record.torchNpuRecord = torchNpuRecord;
    std::string str = Serialize(head, record);

    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody();
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.device_type, torchNpuRecord.memoryUsage.device_type);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.device_index, torchNpuRecord.memoryUsage.device_index);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.data_type, torchNpuRecord.memoryUsage.data_type);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.allocator_type, torchNpuRecord.memoryUsage.allocator_type);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.ptr, torchNpuRecord.memoryUsage.ptr);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.alloc_size, torchNpuRecord.memoryUsage.alloc_size);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.total_allocated, torchNpuRecord.memoryUsage.total_allocated);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.total_reserved, torchNpuRecord.memoryUsage.total_reserved);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.total_active, torchNpuRecord.memoryUsage.total_active);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.stream_ptr, torchNpuRecord.memoryUsage.stream_ptr);
}

TEST(ProtocolTest, test_protocol_parse_torch_npu_record_min)
{
    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};
    
    auto torchNpuRecord = TorchNpuRecord{};
    torchNpuRecord.memoryUsage = MemoryUsage{};
    torchNpuRecord.memoryUsage.device_type = -128;
    torchNpuRecord.memoryUsage.device_index = -128;
    torchNpuRecord.memoryUsage.data_type = 0;
    torchNpuRecord.memoryUsage.allocator_type = 0;
    torchNpuRecord.memoryUsage.ptr = -9223372036854775808;
    torchNpuRecord.memoryUsage.alloc_size = -9223372036854775808;
    torchNpuRecord.memoryUsage.total_allocated = -9223372036854775808;
    torchNpuRecord.memoryUsage.total_reserved = -9223372036854775808;
    torchNpuRecord.memoryUsage.total_active = -9223372036854775808;
    torchNpuRecord.memoryUsage.stream_ptr = -9223372036854775808;

    record.record.torchNpuRecord = torchNpuRecord;
    std::string str = Serialize(head, record);

    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody();
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.device_type, torchNpuRecord.memoryUsage.device_type);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.device_index, torchNpuRecord.memoryUsage.device_index);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.data_type, torchNpuRecord.memoryUsage.data_type);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.allocator_type, torchNpuRecord.memoryUsage.allocator_type);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.ptr, torchNpuRecord.memoryUsage.ptr);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.alloc_size, torchNpuRecord.memoryUsage.alloc_size);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.total_allocated, torchNpuRecord.memoryUsage.total_allocated);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.total_reserved, torchNpuRecord.memoryUsage.total_reserved);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.total_active, torchNpuRecord.memoryUsage.total_active);
    ASSERT_EQ(body.record.torchNpuRecord.memoryUsage.stream_ptr, torchNpuRecord.memoryUsage.stream_ptr);
}

TEST(ProtocolTest, test_protocol_parse_kernerLaunch_Normal_record)
{
    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};
    
    auto kernelLaunchRecord = KernelLaunchRecord{};
    kernelLaunchRecord.recordIndex = 1345;
    kernelLaunchRecord.type = KernelLaunchType::NORMAL;


    record.record.kernelLaunchRecord = kernelLaunchRecord;
    std::string str = Serialize(head, record);

    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody();
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.kernelLaunchRecord.recordIndex, kernelLaunchRecord.recordIndex);
    ASSERT_EQ(body.record.kernelLaunchRecord.type, kernelLaunchRecord.type);
}

TEST(ProtocolTest, test_protocol_parse_kernerLaunch_HandleV2_record)
{
    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};
    
    auto kernelLaunchRecord = KernelLaunchRecord{};
    kernelLaunchRecord.recordIndex = 1345;
    kernelLaunchRecord.type = KernelLaunchType::HANDLEV2;


    record.record.kernelLaunchRecord = kernelLaunchRecord;
    std::string str = Serialize(head, record);

    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody();
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.kernelLaunchRecord.recordIndex, kernelLaunchRecord.recordIndex);
    ASSERT_EQ(body.record.kernelLaunchRecord.type, kernelLaunchRecord.type);
}

TEST(ProtocolTest, test_protocol_parse_kernerLaunch_FlagV2_record)
{
    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};
    
    auto kernelLaunchRecord = KernelLaunchRecord{};
    kernelLaunchRecord.recordIndex = 1345;
    kernelLaunchRecord.type = KernelLaunchType::FLAGV2;


    record.record.kernelLaunchRecord = kernelLaunchRecord;
    std::string str = Serialize(head, record);

    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody();
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.kernelLaunchRecord.recordIndex, kernelLaunchRecord.recordIndex);
    ASSERT_EQ(body.record.kernelLaunchRecord.type, kernelLaunchRecord.type);
}

TEST(ProtocolTest, test_protocol_parse_mstx_MarkA_record)
{
    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};
    
    auto mstxRecord = MstxRecord{};
    mstxRecord.rangeId = 1;
    mstxRecord.markType = MarkType::MARK_A;


    record.record.mstxRecord = mstxRecord;
    std::string str = Serialize(head, record);

    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody();
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.mstxRecord.rangeId, mstxRecord.rangeId);
    ASSERT_EQ(body.record.mstxRecord.markType, mstxRecord.markType);
}

TEST(ProtocolTest, test_protocol_parse_mstx_Start_record)
{
    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};
    
    auto mstxRecord = MstxRecord{};
    mstxRecord.rangeId = 1;
    mstxRecord.markType = MarkType::RANGE_START_A;


    record.record.mstxRecord = mstxRecord;
    std::string str = Serialize(head, record);

    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody();
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.mstxRecord.rangeId, mstxRecord.rangeId);
    ASSERT_EQ(body.record.mstxRecord.markType, mstxRecord.markType);
}

TEST(ProtocolTest, test_protocol_parse_mstx_End_record)
{
    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};
    
    auto mstxRecord = MstxRecord{};
    mstxRecord.rangeId = 1;
    mstxRecord.markType = MarkType::RANGE_END;


    record.record.mstxRecord = mstxRecord;
    std::string str = Serialize(head, record);

    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody();
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.mstxRecord.rangeId, mstxRecord.rangeId);
    ASSERT_EQ(body.record.mstxRecord.markType, mstxRecord.markType);
}
TEST(ProtocolTest, test_protocol_parse_memrecord_max)
{
    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};
    
    auto memOpRecord = MemOpRecord {};
    memOpRecord.recordIndex = 18446744073709551615;
    memOpRecord.kernelIndex = 18446744073709551615;
    memOpRecord.modid = 2147483647;
    memOpRecord.flag = 18446744073709551615;
    memOpRecord.pid = 18446744073709551615;
    memOpRecord.tid = 18446744073709551615;
    memOpRecord.devid = 2147483647;
    memOpRecord.memType = MemOpType::MALLOC;
    memOpRecord.space = MemOpSpace::HOST;
    memOpRecord.addr = 18446744073709551615;
    memOpRecord.memSize = 18446744073709551615;
    memOpRecord.timeStamp = 18446744073709551615;
    record.record.memoryRecord = memOpRecord;
    std::string str = Serialize(head, record);

    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody();
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.memoryRecord.recordIndex, memOpRecord.recordIndex);
    ASSERT_EQ(body.record.memoryRecord.addr, memOpRecord.addr);
    ASSERT_EQ(body.record.memoryRecord.memSize, memOpRecord.memSize);
    ASSERT_EQ(body.record.memoryRecord.timeStamp, memOpRecord.timeStamp);
    ASSERT_EQ(body.record.memoryRecord.tid, memOpRecord.tid);
    ASSERT_EQ(body.record.memoryRecord.pid, memOpRecord.pid);
    ASSERT_EQ(body.record.memoryRecord.flag, memOpRecord.flag);
    ASSERT_EQ(body.record.memoryRecord.devid, memOpRecord.devid);
    ASSERT_EQ(body.record.memoryRecord.memType, memOpRecord.memType);
    ASSERT_EQ(body.record.memoryRecord.space, memOpRecord.space);
    ASSERT_EQ(body.record.memoryRecord.kernelIndex, memOpRecord.kernelIndex);
    ASSERT_EQ(body.record.memoryRecord.modid, memOpRecord.modid);
}
TEST(ProtocolTest, test_protocol_parse_memrecord_min)
{
    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};
    
    auto memOpRecord = MemOpRecord {};
    memOpRecord.recordIndex = 0;
    memOpRecord.kernelIndex = 0;
    memOpRecord.modid = -2147483648;
    memOpRecord.flag = 0;
    memOpRecord.pid = 0;
    memOpRecord.tid = 0;
    memOpRecord.devid = -2147483648;
    memOpRecord.memType = MemOpType::MALLOC;
    memOpRecord.space = MemOpSpace::HOST;
    memOpRecord.addr = 0;
    memOpRecord.memSize = 0;
    memOpRecord.timeStamp = 0;
    record.record.memoryRecord = memOpRecord;
    std::string str = Serialize(head, record);

    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody();
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.memoryRecord.recordIndex, memOpRecord.recordIndex);
    ASSERT_EQ(body.record.memoryRecord.addr, memOpRecord.addr);
    ASSERT_EQ(body.record.memoryRecord.memSize, memOpRecord.memSize);
    ASSERT_EQ(body.record.memoryRecord.timeStamp, memOpRecord.timeStamp);
    ASSERT_EQ(body.record.memoryRecord.tid, memOpRecord.tid);
    ASSERT_EQ(body.record.memoryRecord.pid, memOpRecord.pid);
    ASSERT_EQ(body.record.memoryRecord.flag, memOpRecord.flag);
    ASSERT_EQ(body.record.memoryRecord.devid, memOpRecord.devid);
    ASSERT_EQ(body.record.memoryRecord.memType, memOpRecord.memType);
    ASSERT_EQ(body.record.memoryRecord.space, memOpRecord.space);
    ASSERT_EQ(body.record.memoryRecord.kernelIndex, memOpRecord.kernelIndex);
    ASSERT_EQ(body.record.memoryRecord.modid, memOpRecord.modid);
}

TEST(ProtocolTest, test_protocol_parse_kernellaunchrecord_min)
{
    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};
    
    auto kernelLaunchRecord = KernelLaunchRecord {};
    kernelLaunchRecord.recordIndex = 0;
    kernelLaunchRecord.kernelLaunchIndex = 0;
    kernelLaunchRecord.pid = 0;
    kernelLaunchRecord.tid = 0;
    kernelLaunchRecord.type = KernelLaunchType::NORMAL;
    kernelLaunchRecord.streamId = -2147483648;
    kernelLaunchRecord.blockDim = 0;
    kernelLaunchRecord.timeStamp = 0;
    record.record.kernelLaunchRecord = kernelLaunchRecord;
    std::string str = Serialize(head, record);

    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody();
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.kernelLaunchRecord.recordIndex, kernelLaunchRecord.recordIndex);
    ASSERT_EQ(body.record.kernelLaunchRecord.kernelLaunchIndex, kernelLaunchRecord.kernelLaunchIndex);
    ASSERT_EQ(body.record.kernelLaunchRecord.pid, kernelLaunchRecord.pid);
    ASSERT_EQ(body.record.kernelLaunchRecord.tid, kernelLaunchRecord.tid);
    ASSERT_EQ(body.record.kernelLaunchRecord.type, kernelLaunchRecord.type);
    ASSERT_EQ(body.record.kernelLaunchRecord.streamId, kernelLaunchRecord.streamId);
    ASSERT_EQ(body.record.kernelLaunchRecord.blockDim, kernelLaunchRecord.blockDim);
    ASSERT_EQ(body.record.kernelLaunchRecord.timeStamp, kernelLaunchRecord.timeStamp);
}

TEST(ProtocolTest, test_protocol_parse_kernellaunchrecord_max)
{
    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};
    
    auto kernelLaunchRecord = KernelLaunchRecord {};
    kernelLaunchRecord.recordIndex = 18446744073709551615;
    kernelLaunchRecord.kernelLaunchIndex = 18446744073709551615;
    kernelLaunchRecord.pid = 18446744073709551615;
    kernelLaunchRecord.tid = 18446744073709551615;
    kernelLaunchRecord.type = KernelLaunchType::NORMAL;
    kernelLaunchRecord.streamId = 2147483647;
    kernelLaunchRecord.blockDim = 4294967295;
    kernelLaunchRecord.timeStamp = 18446744073709551615;
    record.record.kernelLaunchRecord = kernelLaunchRecord;
    std::string str = Serialize(head, record);

    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody();
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.kernelLaunchRecord.recordIndex, kernelLaunchRecord.recordIndex);
    ASSERT_EQ(body.record.kernelLaunchRecord.kernelLaunchIndex, kernelLaunchRecord.kernelLaunchIndex);
    ASSERT_EQ(body.record.kernelLaunchRecord.pid, kernelLaunchRecord.pid);
    ASSERT_EQ(body.record.kernelLaunchRecord.tid, kernelLaunchRecord.tid);
    ASSERT_EQ(body.record.kernelLaunchRecord.type, kernelLaunchRecord.type);
    ASSERT_EQ(body.record.kernelLaunchRecord.streamId, kernelLaunchRecord.streamId);
    ASSERT_EQ(body.record.kernelLaunchRecord.blockDim, kernelLaunchRecord.blockDim);
    ASSERT_EQ(body.record.kernelLaunchRecord.timeStamp, kernelLaunchRecord.timeStamp);
}

TEST(ProtocolTest, test_protocol_parse_aclitfrecord_max)
{
    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};
    
    auto aclItfRecord= AclItfRecord {};
    aclItfRecord.recordIndex = 18446744073709551615;
    aclItfRecord.aclItfRecord = 18446744073709551615;
    aclItfRecord.pid = 18446744073709551615;
    aclItfRecord.tid = 18446744073709551615;
    aclItfRecord.type = AclOpType::INIT;
    aclItfRecord.timeStamp = 18446744073709551615;
    record.record.aclItfRecord = aclItfRecord;
    std::string str = Serialize(head, record);

    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody();
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.aclItfRecord.recordIndex, aclItfRecord.recordIndex);
    ASSERT_EQ(body.record.aclItfRecord.aclItfRecord, aclItfRecord.aclItfRecord);
    ASSERT_EQ(body.record.aclItfRecord.pid, aclItfRecord.pid);
    ASSERT_EQ(body.record.aclItfRecord.tid, aclItfRecord.tid);
    ASSERT_EQ(body.record.aclItfRecord.type, aclItfRecord.type);
    ASSERT_EQ(body.record.aclItfRecord.timeStamp, aclItfRecord.timeStamp);
}

TEST(ProtocolTest, test_protocol_parse_aclitfrecord_min)
{
    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};
    
    auto aclItfRecord= AclItfRecord {};
    aclItfRecord.recordIndex = 0;
    aclItfRecord.aclItfRecord = 0;
    aclItfRecord.pid = 0;
    aclItfRecord.tid = 0;
    aclItfRecord.type = AclOpType::INIT;
    aclItfRecord.timeStamp = 0;
    record.record.aclItfRecord = aclItfRecord;
    std::string str = Serialize(head, record);

    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody();
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.aclItfRecord.recordIndex, aclItfRecord.recordIndex);
    ASSERT_EQ(body.record.aclItfRecord.aclItfRecord, aclItfRecord.aclItfRecord);
    ASSERT_EQ(body.record.aclItfRecord.pid, aclItfRecord.pid);
    ASSERT_EQ(body.record.aclItfRecord.tid, aclItfRecord.tid);
    ASSERT_EQ(body.record.aclItfRecord.type, aclItfRecord.type);
    ASSERT_EQ(body.record.aclItfRecord.timeStamp, aclItfRecord.timeStamp);
}

TEST(ProtocolTest, test_protocol_parse_device_record)
{
    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};

    auto memOpRecord = MemOpRecord {};
    memOpRecord.space = MemOpSpace::DEVICE;
    record.record.memoryRecord = memOpRecord;
    std::string str = Serialize(head, record);

    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody();
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.memoryRecord.space, memOpRecord.space);
}