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
    PacketHead head {PacketType::RECORD, sizeof(EventRecord)};
    auto record = EventRecord {};
    auto memRecord = MemOpRecord {};
    memRecord.recordIndex = 123;
    memRecord.kernelIndex = 123;
    memRecord.flag = 123;
    memRecord.pid = 345;
    memRecord.tid = 345;
    memRecord.devId = 9;
    memRecord.subtype = RecordSubType::MALLOC;
    memRecord.space = MemOpSpace::HOST;
    memRecord.modid = 234;
    memRecord.addr = 0x7958;
    memRecord.memSize = 1024;
    memRecord.timestamp = 1234567;
    record.type = RecordType::MEMORY_RECORD;
    record.record.memoryRecord = memRecord;
    std::string str = Serialize(head, record);
    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();
    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody().record.eventRecord;
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.memoryRecord.recordIndex, memRecord.recordIndex);
    ASSERT_EQ(body.record.memoryRecord.addr, memRecord.addr);
    ASSERT_EQ(body.record.memoryRecord.memSize, memRecord.memSize);
    ASSERT_EQ(body.record.memoryRecord.timestamp, memRecord.timestamp);
    ASSERT_EQ(body.record.memoryRecord.kernelIndex, memRecord.kernelIndex);
    ASSERT_EQ(body.record.memoryRecord.flag, memRecord.flag);
    ASSERT_EQ(body.record.memoryRecord.pid, memRecord.pid);
    ASSERT_EQ(body.record.memoryRecord.tid, memRecord.tid);
    ASSERT_EQ(body.record.memoryRecord.devId, memRecord.devId);
    ASSERT_EQ(body.record.memoryRecord.subtype, memRecord.subtype);
    ASSERT_EQ(body.record.memoryRecord.space, memRecord.space);
    ASSERT_EQ(body.record.memoryRecord.modid, memRecord.modid);
}

TEST(ProtocolTest, test_protocol_parse_Invalidrecord)
{
    PacketHead head {PacketType::RECORD, sizeof(EventRecord)};
    auto record = EventRecord{};
    std::string str = Serialize(head, record);
    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_FALSE(result.GetPacketHead().type == PacketType::INVALID);
}

TEST(ProtocolTest, test_protocol_parse_acl_itf_Init_record)
{
    PacketHead head {PacketType::RECORD, sizeof(EventRecord)};
    auto record = EventRecord {};
    auto aclItfRecord = AclItfRecord {};
    aclItfRecord.recordIndex = 13456;
    aclItfRecord.pid = 123;
    aclItfRecord.tid = 123;
    aclItfRecord.aclItfRecordIndex = 123;
    aclItfRecord.timestamp = 234;
    aclItfRecord.subtype = RecordSubType::INIT;
    record.type = RecordType::ACL_ITF_RECORD;
    record.record.aclItfRecord = aclItfRecord;
    std::string str = Serialize(head, record);
    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody().record.eventRecord;
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.aclItfRecord.recordIndex, aclItfRecord.recordIndex);
    ASSERT_EQ(body.record.aclItfRecord.type, aclItfRecord.type);
    ASSERT_EQ(body.record.aclItfRecord.pid, aclItfRecord.pid);
    ASSERT_EQ(body.record.aclItfRecord.tid, aclItfRecord.tid);
    ASSERT_EQ(body.record.aclItfRecord.aclItfRecordIndex, aclItfRecord.aclItfRecordIndex);
    ASSERT_EQ(body.record.aclItfRecord.timestamp, aclItfRecord.timestamp);
}

TEST(ProtocolTest, test_protocol_parse_acl_itf_finalize_record)
{
    PacketHead head {PacketType::RECORD, sizeof(EventRecord)};
    auto record = EventRecord {};
    auto aclItfRecord = AclItfRecord {};
    aclItfRecord.recordIndex = 13456;
    aclItfRecord.pid = 123;
    aclItfRecord.tid = 123;
    aclItfRecord.aclItfRecordIndex = 123;
    aclItfRecord.timestamp = 234;
    aclItfRecord.subtype = RecordSubType::FINALIZE;
    record.type = RecordType::ACL_ITF_RECORD;
    record.record.aclItfRecord = aclItfRecord;
    std::string str = Serialize(head, record);
    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody().record.eventRecord;
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.aclItfRecord.recordIndex, aclItfRecord.recordIndex);
    ASSERT_EQ(body.record.aclItfRecord.type, aclItfRecord.type);
    ASSERT_EQ(body.record.aclItfRecord.pid, aclItfRecord.pid);
    ASSERT_EQ(body.record.aclItfRecord.tid, aclItfRecord.tid);
    ASSERT_EQ(body.record.aclItfRecord.aclItfRecordIndex, aclItfRecord.aclItfRecordIndex);
    ASSERT_EQ(body.record.aclItfRecord.timestamp, aclItfRecord.timestamp);
}

TEST(ProtocolTest, test_protocol_parse_torch_npu_record)
{
    PacketHead head {PacketType::RECORD, sizeof(EventRecord)};
    auto record = EventRecord {};
    
    auto memPoolRecord = MemPoolRecord{};
    memPoolRecord.memoryUsage = MemoryUsage{};
    memPoolRecord.memoryUsage.deviceType = 1;
    memPoolRecord.memoryUsage.deviceIndex = 2;
    memPoolRecord.memoryUsage.dataType = 3;
    memPoolRecord.memoryUsage.allocatorType = 4;
    memPoolRecord.memoryUsage.ptr = 5;
    memPoolRecord.memoryUsage.allocSize = 6;
    memPoolRecord.memoryUsage.totalAllocated = 7;
    memPoolRecord.memoryUsage.totalReserved = 8;
    memPoolRecord.memoryUsage.totalActive = 9;
    memPoolRecord.memoryUsage.streamPtr = 10;

    record.record.memPoolRecord = memPoolRecord;
    std::string str = Serialize(head, record);
    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody().record.eventRecord;
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.deviceType, memPoolRecord.memoryUsage.deviceType);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.deviceIndex, memPoolRecord.memoryUsage.deviceIndex);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.dataType, memPoolRecord.memoryUsage.dataType);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.allocatorType, memPoolRecord.memoryUsage.allocatorType);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.ptr, memPoolRecord.memoryUsage.ptr);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.allocSize, memPoolRecord.memoryUsage.allocSize);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.totalAllocated, memPoolRecord.memoryUsage.totalAllocated);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.totalReserved, memPoolRecord.memoryUsage.totalReserved);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.totalActive, memPoolRecord.memoryUsage.totalActive);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.streamPtr, memPoolRecord.memoryUsage.streamPtr);
}

TEST(ProtocolTest, test_protocol_parse_torch_npu_record_max)
{
    PacketHead head {PacketType::RECORD, sizeof(EventRecord)};
    auto record = EventRecord {};
    
    auto memPoolRecord = MemPoolRecord{};
    memPoolRecord.memoryUsage = MemoryUsage{};
    memPoolRecord.memoryUsage.deviceType = 127;
    memPoolRecord.memoryUsage.deviceIndex = 127;
    memPoolRecord.memoryUsage.dataType = 255;
    memPoolRecord.memoryUsage.allocatorType = 255;
    memPoolRecord.memoryUsage.ptr = 9223372036854775807;
    memPoolRecord.memoryUsage.allocSize = 9223372036854775807;
    memPoolRecord.memoryUsage.totalAllocated = 9223372036854775807;
    memPoolRecord.memoryUsage.totalReserved = 9223372036854775807;
    memPoolRecord.memoryUsage.totalActive = 9223372036854775807;
    memPoolRecord.memoryUsage.streamPtr = 9223372036854775807;

    record.record.memPoolRecord = memPoolRecord;
    std::string str = Serialize(head, record);
    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody().record.eventRecord;
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.deviceType, memPoolRecord.memoryUsage.deviceType);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.deviceIndex, memPoolRecord.memoryUsage.deviceIndex);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.dataType, memPoolRecord.memoryUsage.dataType);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.allocatorType, memPoolRecord.memoryUsage.allocatorType);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.ptr, memPoolRecord.memoryUsage.ptr);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.allocSize, memPoolRecord.memoryUsage.allocSize);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.totalAllocated, memPoolRecord.memoryUsage.totalAllocated);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.totalReserved, memPoolRecord.memoryUsage.totalReserved);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.totalActive, memPoolRecord.memoryUsage.totalActive);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.streamPtr, memPoolRecord.memoryUsage.streamPtr);
}

TEST(ProtocolTest, test_protocol_parse_torch_npu_record_min)
{
    auto record = EventRecord {};
    
    auto memPoolRecord = MemPoolRecord{};
    memPoolRecord.memoryUsage = MemoryUsage{};
    memPoolRecord.memoryUsage.deviceType = -128;
    memPoolRecord.memoryUsage.deviceIndex = -128;
    memPoolRecord.memoryUsage.dataType = 0;
    memPoolRecord.memoryUsage.allocatorType = 0;
    memPoolRecord.memoryUsage.ptr = -9223372036854775808;
    memPoolRecord.memoryUsage.allocSize = -9223372036854775808;
    memPoolRecord.memoryUsage.totalAllocated = -9223372036854775808;
    memPoolRecord.memoryUsage.totalReserved = -9223372036854775808;
    memPoolRecord.memoryUsage.totalActive = -9223372036854775808;
    memPoolRecord.memoryUsage.streamPtr = -9223372036854775808;

    record.record.memPoolRecord = memPoolRecord;
    std::string testMsg = "test";
    record.pyStackLen = testMsg.size();
    record.cStackLen = testMsg.size();
    PacketHead head {PacketType::RECORD, sizeof(EventRecord) + record.pyStackLen + record.cStackLen};
    std::string str = Serialize(head, record);
    str += testMsg + testMsg;
    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody().record.eventRecord;
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.deviceType, memPoolRecord.memoryUsage.deviceType);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.deviceIndex, memPoolRecord.memoryUsage.deviceIndex);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.dataType, memPoolRecord.memoryUsage.dataType);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.allocatorType, memPoolRecord.memoryUsage.allocatorType);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.ptr, memPoolRecord.memoryUsage.ptr);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.allocSize, memPoolRecord.memoryUsage.allocSize);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.totalAllocated, memPoolRecord.memoryUsage.totalAllocated);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.totalReserved, memPoolRecord.memoryUsage.totalReserved);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.totalActive, memPoolRecord.memoryUsage.totalActive);
    ASSERT_EQ(body.record.memPoolRecord.memoryUsage.streamPtr, memPoolRecord.memoryUsage.streamPtr);
}

TEST(ProtocolTest, test_protocol_parse_kernerLaunch_Normal_record)
{
    auto record = EventRecord {};
    
    auto kernelLaunchRecord = KernelLaunchRecord{};
    kernelLaunchRecord.recordIndex = 1345;
    kernelLaunchRecord.subtype = RecordSubType::NORMAL;

    record.record.kernelLaunchRecord = kernelLaunchRecord;
    std::string testMsg = "test";
    record.pyStackLen = testMsg.size();
    record.cStackLen = testMsg.size();
    PacketHead head {PacketType::RECORD, sizeof(EventRecord) + record.pyStackLen + record.cStackLen};
    std::string str = Serialize(head, record);
    str += testMsg + testMsg;
    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody().record.eventRecord;
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.kernelLaunchRecord.recordIndex, kernelLaunchRecord.recordIndex);
    ASSERT_EQ(body.record.kernelLaunchRecord.type, kernelLaunchRecord.type);
}

TEST(ProtocolTest, test_protocol_parse_kernerLaunch_HandleV2_record)
{
    auto record = EventRecord {};
    
    auto kernelLaunchRecord = KernelLaunchRecord{};
    kernelLaunchRecord.recordIndex = 1345;
    kernelLaunchRecord.subtype = RecordSubType::HANDLEV2;

    record.record.kernelLaunchRecord = kernelLaunchRecord;
    std::string testMsg = "test";
    record.pyStackLen = testMsg.size();
    record.cStackLen = testMsg.size();
    PacketHead head {PacketType::RECORD, sizeof(EventRecord) + record.pyStackLen + record.cStackLen};
    std::string str = Serialize(head, record);
    str += testMsg + testMsg;
    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody().record.eventRecord;
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.kernelLaunchRecord.recordIndex, kernelLaunchRecord.recordIndex);
    ASSERT_EQ(body.record.kernelLaunchRecord.type, kernelLaunchRecord.type);
}

TEST(ProtocolTest, test_protocol_parse_kernerLaunch_FlagV2_record)
{
    auto record = EventRecord {};
    
    auto kernelLaunchRecord = KernelLaunchRecord{};
    kernelLaunchRecord.recordIndex = 1345;
    kernelLaunchRecord.subtype = RecordSubType::FLAGV2;

    record.record.kernelLaunchRecord = kernelLaunchRecord;
    std::string testMsg = "test";
    record.pyStackLen = testMsg.size();
    record.cStackLen = testMsg.size();
    PacketHead head {PacketType::RECORD, sizeof(EventRecord) + record.pyStackLen + record.cStackLen};
    std::string str = Serialize(head, record);
    str += testMsg + testMsg;
    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody().record.eventRecord;
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.kernelLaunchRecord.recordIndex, kernelLaunchRecord.recordIndex);
    ASSERT_EQ(body.record.kernelLaunchRecord.type, kernelLaunchRecord.type);
}

TEST(ProtocolTest, test_protocol_parse_mstx_MarkA_record)
{
    auto record = EventRecord {};
    
    auto mstxRecord = MstxRecord{};
    mstxRecord.rangeId = 0;
    mstxRecord.stepId = 1;
    mstxRecord.markType = MarkType::MARK_A;


    record.record.mstxRecord = mstxRecord;
    std::string testMsg = "test";
    record.pyStackLen = testMsg.size();
    record.cStackLen = testMsg.size();
    PacketHead head {PacketType::RECORD, sizeof(EventRecord) + record.pyStackLen + record.cStackLen};
    std::string str = Serialize(head, record);
    str += testMsg + testMsg;
    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody().record.eventRecord;
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.mstxRecord.rangeId, mstxRecord.rangeId);
    ASSERT_EQ(body.record.mstxRecord.stepId, mstxRecord.stepId);
    ASSERT_EQ(body.record.mstxRecord.markType, mstxRecord.markType);
}

TEST(ProtocolTest, test_protocol_parse_mstx_Start_record)
{
    auto record = EventRecord {};
    
    auto mstxRecord = MstxRecord{};
    mstxRecord.rangeId = 0;
    mstxRecord.stepId = 1;
    mstxRecord.markType = MarkType::RANGE_START_A;


    record.record.mstxRecord = mstxRecord;
    std::string testMsg = "test";
    record.pyStackLen = testMsg.size();
    record.cStackLen = testMsg.size();
    PacketHead head {PacketType::RECORD, sizeof(record) + record.pyStackLen + record.cStackLen};
    std::string str = Serialize(head, record);
    str += testMsg + testMsg;
    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody().record.eventRecord;
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.mstxRecord.rangeId, mstxRecord.rangeId);
    ASSERT_EQ(body.record.mstxRecord.stepId, mstxRecord.stepId);
    ASSERT_EQ(body.record.mstxRecord.markType, mstxRecord.markType);
}

TEST(ProtocolTest, test_protocol_parse_mstx_End_record)
{
    auto record = EventRecord {};
    
    auto mstxRecord = MstxRecord{};
    mstxRecord.rangeId = 0;
    mstxRecord.stepId = 1;
    mstxRecord.markType = MarkType::RANGE_END;

    record.record.mstxRecord = mstxRecord;
    std::string testMsg = "test";
    record.pyStackLen = testMsg.size();
    record.cStackLen = testMsg.size();
    PacketHead head {PacketType::RECORD, sizeof(record) + record.pyStackLen + record.cStackLen};
    std::string str = Serialize(head, record);
    str += testMsg + testMsg;
    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody().record.eventRecord;
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.mstxRecord.rangeId, mstxRecord.rangeId);
    ASSERT_EQ(body.record.mstxRecord.stepId, mstxRecord.stepId);
    ASSERT_EQ(body.record.mstxRecord.markType, mstxRecord.markType);
}
TEST(ProtocolTest, test_protocol_parse_memrecord_max)
{
    auto record = EventRecord {};
    
    auto memOpRecord = MemOpRecord {};
    memOpRecord.recordIndex = 18446744073709551615;
    memOpRecord.kernelIndex = 18446744073709551615;
    memOpRecord.modid = 2147483647;
    memOpRecord.flag = 18446744073709551615;
    memOpRecord.pid = 18446744073709551615;
    memOpRecord.tid = 18446744073709551615;
    memOpRecord.devId = 2147483647;
    memOpRecord.subtype = RecordSubType::MALLOC;
    memOpRecord.space = MemOpSpace::HOST;
    memOpRecord.addr = 18446744073709551615;
    memOpRecord.memSize = 18446744073709551615;
    memOpRecord.timestamp = 18446744073709551615;
    record.record.memoryRecord = memOpRecord;
    std::string testMsg = "test";
    record.pyStackLen = testMsg.size();
    record.cStackLen = testMsg.size();
    PacketHead head {PacketType::RECORD, sizeof(record) + record.pyStackLen + record.cStackLen};
    std::string str = Serialize(head, record);
    str += testMsg + testMsg;
    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody().record.eventRecord;
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.memoryRecord.recordIndex, memOpRecord.recordIndex);
    ASSERT_EQ(body.record.memoryRecord.addr, memOpRecord.addr);
    ASSERT_EQ(body.record.memoryRecord.memSize, memOpRecord.memSize);
    ASSERT_EQ(body.record.memoryRecord.timestamp, memOpRecord.timestamp);
    ASSERT_EQ(body.record.memoryRecord.tid, memOpRecord.tid);
    ASSERT_EQ(body.record.memoryRecord.pid, memOpRecord.pid);
    ASSERT_EQ(body.record.memoryRecord.flag, memOpRecord.flag);
    ASSERT_EQ(body.record.memoryRecord.devId, memOpRecord.devId);
    ASSERT_EQ(body.record.memoryRecord.subtype, memOpRecord.subtype);
    ASSERT_EQ(body.record.memoryRecord.space, memOpRecord.space);
    ASSERT_EQ(body.record.memoryRecord.kernelIndex, memOpRecord.kernelIndex);
    ASSERT_EQ(body.record.memoryRecord.modid, memOpRecord.modid);
}
TEST(ProtocolTest, test_protocol_parse_memrecord_min)
{
    auto record = EventRecord {};
    
    auto memOpRecord = MemOpRecord {};
    memOpRecord.recordIndex = 0;
    memOpRecord.kernelIndex = 0;
    memOpRecord.modid = -2147483648;
    memOpRecord.flag = 0;
    memOpRecord.pid = 0;
    memOpRecord.tid = 0;
    memOpRecord.devId = -2147483648;
    memOpRecord.subtype = RecordSubType::MALLOC;
    memOpRecord.space = MemOpSpace::HOST;
    memOpRecord.addr = 0;
    memOpRecord.memSize = 0;
    memOpRecord.timestamp = 0;
    record.record.memoryRecord = memOpRecord;
    std::string testMsg = "test";
    record.pyStackLen = testMsg.size();
    record.cStackLen = testMsg.size();
    PacketHead head {PacketType::RECORD, sizeof(record) + record.pyStackLen + record.cStackLen};
    std::string str = Serialize(head, record);
    str += testMsg + testMsg;
    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody().record.eventRecord;
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.memoryRecord.recordIndex, memOpRecord.recordIndex);
    ASSERT_EQ(body.record.memoryRecord.addr, memOpRecord.addr);
    ASSERT_EQ(body.record.memoryRecord.memSize, memOpRecord.memSize);
    ASSERT_EQ(body.record.memoryRecord.timestamp, memOpRecord.timestamp);
    ASSERT_EQ(body.record.memoryRecord.tid, memOpRecord.tid);
    ASSERT_EQ(body.record.memoryRecord.pid, memOpRecord.pid);
    ASSERT_EQ(body.record.memoryRecord.flag, memOpRecord.flag);
    ASSERT_EQ(body.record.memoryRecord.devId, memOpRecord.devId);
    ASSERT_EQ(body.record.memoryRecord.subtype, memOpRecord.subtype);
    ASSERT_EQ(body.record.memoryRecord.space, memOpRecord.space);
    ASSERT_EQ(body.record.memoryRecord.kernelIndex, memOpRecord.kernelIndex);
    ASSERT_EQ(body.record.memoryRecord.modid, memOpRecord.modid);
}

TEST(ProtocolTest, test_protocol_parse_kernellaunchrecord_min)
{
    auto record = EventRecord {};
    
    auto kernelLaunchRecord = KernelLaunchRecord {};
    kernelLaunchRecord.recordIndex = 0;
    kernelLaunchRecord.kernelLaunchIndex = 0;
    kernelLaunchRecord.pid = 0;
    kernelLaunchRecord.tid = 0;
    kernelLaunchRecord.subtype = RecordSubType::NORMAL;
    kernelLaunchRecord.streamId = -2147483648;
    kernelLaunchRecord.blockDim = 0;
    kernelLaunchRecord.timestamp = 0;
    record.record.kernelLaunchRecord = kernelLaunchRecord;
    std::string testMsg = "test";
    record.pyStackLen = testMsg.size();
    record.cStackLen = testMsg.size();
    PacketHead head {PacketType::RECORD, sizeof(record) + record.pyStackLen + record.cStackLen};
    std::string str = Serialize(head, record);
    str += testMsg + testMsg;
    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody().record.eventRecord;
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.kernelLaunchRecord.recordIndex, kernelLaunchRecord.recordIndex);
    ASSERT_EQ(body.record.kernelLaunchRecord.kernelLaunchIndex, kernelLaunchRecord.kernelLaunchIndex);
    ASSERT_EQ(body.record.kernelLaunchRecord.pid, kernelLaunchRecord.pid);
    ASSERT_EQ(body.record.kernelLaunchRecord.tid, kernelLaunchRecord.tid);
    ASSERT_EQ(body.record.kernelLaunchRecord.type, kernelLaunchRecord.type);
    ASSERT_EQ(body.record.kernelLaunchRecord.streamId, kernelLaunchRecord.streamId);
    ASSERT_EQ(body.record.kernelLaunchRecord.blockDim, kernelLaunchRecord.blockDim);
    ASSERT_EQ(body.record.kernelLaunchRecord.timestamp, kernelLaunchRecord.timestamp);
}

TEST(ProtocolTest, test_protocol_parse_kernellaunchrecord_max)
{
    auto record = EventRecord {};
    
    auto kernelLaunchRecord = KernelLaunchRecord {};
    kernelLaunchRecord.recordIndex = 18446744073709551615;
    kernelLaunchRecord.kernelLaunchIndex = 18446744073709551615;
    kernelLaunchRecord.pid = 18446744073709551615;
    kernelLaunchRecord.tid = 18446744073709551615;
    kernelLaunchRecord.subtype = RecordSubType::NORMAL;
    kernelLaunchRecord.streamId = 2147483647;
    kernelLaunchRecord.blockDim = 4294967295;
    kernelLaunchRecord.timestamp = 18446744073709551615;
    record.record.kernelLaunchRecord = kernelLaunchRecord;
    std::string testMsg = "test";
    record.pyStackLen = testMsg.size();
    record.cStackLen = testMsg.size();
    PacketHead head {PacketType::RECORD, sizeof(record) + record.pyStackLen + record.cStackLen};
    std::string str = Serialize(head, record);
    str += testMsg + testMsg;
    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody().record.eventRecord;
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.kernelLaunchRecord.recordIndex, kernelLaunchRecord.recordIndex);
    ASSERT_EQ(body.record.kernelLaunchRecord.kernelLaunchIndex, kernelLaunchRecord.kernelLaunchIndex);
    ASSERT_EQ(body.record.kernelLaunchRecord.pid, kernelLaunchRecord.pid);
    ASSERT_EQ(body.record.kernelLaunchRecord.tid, kernelLaunchRecord.tid);
    ASSERT_EQ(body.record.kernelLaunchRecord.type, kernelLaunchRecord.type);
    ASSERT_EQ(body.record.kernelLaunchRecord.streamId, kernelLaunchRecord.streamId);
    ASSERT_EQ(body.record.kernelLaunchRecord.blockDim, kernelLaunchRecord.blockDim);
    ASSERT_EQ(body.record.kernelLaunchRecord.timestamp, kernelLaunchRecord.timestamp);
}

TEST(ProtocolTest, test_protocol_parse_aclitfrecord_max)
{
    auto record = EventRecord {};
    
    auto aclItfRecord= AclItfRecord {};
    aclItfRecord.recordIndex = 18446744073709551615;
    aclItfRecord.aclItfRecordIndex = 18446744073709551615;
    aclItfRecord.pid = 18446744073709551615;
    aclItfRecord.tid = 18446744073709551615;
    aclItfRecord.subtype = RecordSubType::INIT;
    aclItfRecord.timestamp = 18446744073709551615;
    record.record.aclItfRecord = aclItfRecord;
    std::string testMsg = "test";
    record.pyStackLen = testMsg.size();
    record.cStackLen = testMsg.size();
    PacketHead head {PacketType::RECORD, sizeof(record) + record.pyStackLen + record.cStackLen};
    std::string str = Serialize(head, record);
    str += testMsg + testMsg;
    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody().record.eventRecord;
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.aclItfRecord.recordIndex, aclItfRecord.recordIndex);
    ASSERT_EQ(body.record.aclItfRecord.aclItfRecordIndex, aclItfRecord.aclItfRecordIndex);
    ASSERT_EQ(body.record.aclItfRecord.pid, aclItfRecord.pid);
    ASSERT_EQ(body.record.aclItfRecord.tid, aclItfRecord.tid);
    ASSERT_EQ(body.record.aclItfRecord.type, aclItfRecord.type);
    ASSERT_EQ(body.record.aclItfRecord.timestamp, aclItfRecord.timestamp);
}

TEST(ProtocolTest, test_protocol_parse_aclitfrecord_min)
{
    auto record = EventRecord {};
    
    auto aclItfRecord= AclItfRecord {};
    aclItfRecord.recordIndex = 0;
    aclItfRecord.aclItfRecordIndex = 0;
    aclItfRecord.pid = 0;
    aclItfRecord.tid = 0;
    aclItfRecord.subtype = RecordSubType::INIT;
    aclItfRecord.timestamp = 0;
    record.record.aclItfRecord = aclItfRecord;
    std::string testMsg = "test";
    record.pyStackLen = testMsg.size();
    record.cStackLen = testMsg.size();
    PacketHead head {PacketType::RECORD, sizeof(record) + record.pyStackLen + record.cStackLen};
    std::string str = Serialize(head, record);
    str += testMsg + testMsg;
    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody().record.eventRecord;
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.aclItfRecord.recordIndex, aclItfRecord.recordIndex);
    ASSERT_EQ(body.record.aclItfRecord.aclItfRecordIndex, aclItfRecord.aclItfRecordIndex);
    ASSERT_EQ(body.record.aclItfRecord.pid, aclItfRecord.pid);
    ASSERT_EQ(body.record.aclItfRecord.tid, aclItfRecord.tid);
    ASSERT_EQ(body.record.aclItfRecord.type, aclItfRecord.type);
    ASSERT_EQ(body.record.aclItfRecord.timestamp, aclItfRecord.timestamp);
}

TEST(ProtocolTest, test_protocol_parse_device_record)
{
    auto record = EventRecord {};

    auto memOpRecord = MemOpRecord {};
    memOpRecord.space = MemOpSpace::DEVICE;
    record.record.memoryRecord = memOpRecord;
    std::string testMsg = "test";
    record.pyStackLen = testMsg.size();
    record.cStackLen = testMsg.size();
    PacketHead head {PacketType::RECORD, sizeof(record) + record.pyStackLen + record.cStackLen};
    std::string str = Serialize(head, record);
    str += testMsg + testMsg;
    Protocol protocol {};
    protocol.Feed(str);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::RECORD);

    auto body = result.GetPacketBody().record.eventRecord;
    ASSERT_EQ(body.type, record.type);
    ASSERT_EQ(body.record.memoryRecord.space, memOpRecord.space);
}

TEST(ProtocolTest, test_protocol_parse_log_record)
{
    std::string logMsg = "test";
    Leaks::PacketHead head {Leaks::PacketType::LOG, logMsg.size()};
    std::string buffer = Leaks::Serialize(head);
    buffer += logMsg;

    Protocol protocol {};
    protocol.Feed(buffer);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::LOG);

    auto body = std::string(result.GetPacketBody().log.buf, result.GetPacketBody().log.buf + 4);
    ASSERT_EQ(body, logMsg);
}

TEST(ProtocolTest, test_protocol_drop_user_bytes)
{
    std::string logMsg(2048, 'a');
    Leaks::PacketHead head {Leaks::PacketType::LOG, logMsg.size()};
    std::string buffer = Leaks::Serialize(head);
    buffer += logMsg;

    Protocol protocol {};
    protocol.Feed(buffer);

    auto result = protocol.GetPacket();

    EXPECT_TRUE(result.GetPacketHead().type == PacketType::LOG);

    auto body = std::string(result.GetPacketBody().log.buf, result.GetPacketBody().log.buf + 2048);
    ASSERT_EQ(body, logMsg);
}