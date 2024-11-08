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

TEST(ProtocolTest, test_protocol_parse_step_record)
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