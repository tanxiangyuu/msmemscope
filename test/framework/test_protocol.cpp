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
}

TEST(ProtocolTest, test_protocol_parse_acl_itf_finalize_record)
{
    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};
    auto aclItfRecord = AclItfRecord {};
    aclItfRecord.recordIndex = 13456;
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