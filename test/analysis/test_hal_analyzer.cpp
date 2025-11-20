// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <string>
#include "hal_analyzer.h"
#include "record_info.h"
#include "config_info.h"
#include "bit_field.h"

using namespace MemScope;

TEST(HalAnalyzerTest, do_hal_record_except_memscope) {
    Config analysisConfig;
    BitField<decltype(analysisConfig.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    analysisConfig.eventType = eventBit.getValue();
    ClientId clientId = 0;
    MemOpRecord record1;
    record1.type = RecordType::MEMORY_RECORD;
    record1.flag = 2377900603261207558;
    record1.recordIndex = 1;
    record1.space = MemOpSpace::DEVICE;
    record1.subtype = RecordSubType::MALLOC;
    record1.addr = 0x7958;
    record1.memSize = 1024;
    record1.timestamp = 1234567;
 
    MemOpRecord record2;
    record2.type = RecordType::MEMORY_RECORD;
    record2.flag = 18374686480754951175;
    record2.recordIndex = 2;
    record2.space = MemOpSpace::INVALID;
    record2.subtype = RecordSubType::MALLOC;
    record2.addr = 0x7957;
    record2.memSize = 512;
    record2.timestamp = 1234568;
 
    MemOpRecord record4;
    record4.type = RecordType::MEMORY_RECORD;
    record4.flag = 504403158275081222;
    record4.recordIndex = 4;
    record4.space = MemOpSpace::DEVICE;
    record4.subtype = RecordSubType::MALLOC;
    record4.addr = 0x7960;
    record4.memSize = 1024;
    record4.timestamp = 1234557;
 
    EXPECT_TRUE(HalAnalyzer::GetInstance(analysisConfig).Record(clientId, record1));
    EXPECT_TRUE(HalAnalyzer::GetInstance(analysisConfig).Record(clientId, record2));
    EXPECT_TRUE(HalAnalyzer::GetInstance(analysisConfig).Record(clientId, record4));
}

TEST(HalAnalyzerTest, do_record_except_no_memscope) {
    Config analysisConfig;
    BitField<decltype(analysisConfig.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    analysisConfig.eventType = eventBit.getValue();
    ClientId clientId = 0;
    MemOpRecord record1;
    record1.type = RecordType::MEMORY_RECORD;
    record1.flag = 2377900603261207558;
    record1.recordIndex = 1;
    record1.space = MemOpSpace::DEVICE;
    record1.subtype = RecordSubType::MALLOC;
    record1.addr = 0x7958;
    record1.memSize = 1024;
    record1.timestamp = 1234567;

    MemOpRecord record3;
    record3.type = RecordType::MEMORY_RECORD;
    record3.recordIndex = 3;
    record3.space = MemOpSpace::INVALID;
    record3.subtype = RecordSubType::FREE;
    record3.addr = 0x7958;
    record3.memSize = 0;

    EXPECT_TRUE(HalAnalyzer::GetInstance(analysisConfig).Record(clientId, record1));
    EXPECT_TRUE(HalAnalyzer::GetInstance(analysisConfig).Record(clientId, record3));
}

TEST(HalAnalyzerTest, do_record_excpet_double_free) {
    Config analysisConfig;
    BitField<decltype(analysisConfig.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    analysisConfig.eventType = eventBit.getValue();
    ClientId clientId = 0;
    MemOpRecord record1;
    record1.type = RecordType::MEMORY_RECORD;
    record1.flag = 2377900603261207558;
    record1.recordIndex = 1;
    record1.space = MemOpSpace::DEVICE;
    record1.subtype = RecordSubType::MALLOC;
    record1.addr = 0x7958;
    record1.memSize = 1024;
    record1.timestamp = 1234567;

    MemOpRecord record2;
    record2.type = RecordType::MEMORY_RECORD;
    record2.recordIndex = 2;
    record2.space = MemOpSpace::INVALID;
    record2.subtype = RecordSubType::FREE;
    record2.addr = 0x7958;
    record2.memSize = 0;

    MemOpRecord record3;
    record3.type = RecordType::MEMORY_RECORD;
    record3.recordIndex = 3;
    record3.space = MemOpSpace::INVALID;
    record3.subtype = RecordSubType::FREE;
    record3.addr = 0x7958;
    record3.memSize = 0;

    EXPECT_TRUE(HalAnalyzer::GetInstance(analysisConfig).Record(clientId, record1));
    EXPECT_TRUE(HalAnalyzer::GetInstance(analysisConfig).Record(clientId, record2));
    EXPECT_TRUE(HalAnalyzer::GetInstance(analysisConfig).Record(clientId, record3));
}

TEST(HalAnalyzerTest, do_record_except_double_malloc) {
    Config analysisConfig;
    BitField<decltype(analysisConfig.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    analysisConfig.eventType = eventBit.getValue();
    ClientId clientId = 0;
    MemOpRecord record1;
    record1.type = RecordType::MEMORY_RECORD;
    record1.flag = 2377900603261207558;
    record1.recordIndex = 1;
    record1.space = MemOpSpace::DEVICE;
    record1.subtype = RecordSubType::MALLOC;
    record1.addr = 0x7958;
    record1.memSize = 1024;
    record1.timestamp = 1234567;

    MemOpRecord record2;
    record2.type = RecordType::MEMORY_RECORD;
    record2.flag = 2377900603261207558;
    record2.recordIndex = 2;
    record2.space = MemOpSpace::DEVICE;
    record2.subtype = RecordSubType::MALLOC;
    record2.addr = 0x7958;
    record2.memSize = 1024;
    record2.timestamp = 1234567;

    EXPECT_TRUE(HalAnalyzer::GetInstance(analysisConfig).Record(clientId, record1));
    EXPECT_TRUE(HalAnalyzer::GetInstance(analysisConfig).Record(clientId, record2));
}

TEST(HalAnalyzerTest, do_record_except_free_null) {
    Config analysisConfig;
    BitField<decltype(analysisConfig.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    analysisConfig.eventType = eventBit.getValue();
    ClientId clientId = 0;
    MemOpRecord record1;
    record1.type = RecordType::MEMORY_RECORD;
    record1.recordIndex = 1;
    record1.space = MemOpSpace::INVALID;
    record1.subtype = RecordSubType::FREE;
    record1.addr = 0x7958;
    record1.memSize = 0;

    EXPECT_TRUE(HalAnalyzer::GetInstance(analysisConfig).Record(clientId, record1));
}

TEST(HalAnalyzerTest, do_record_fail) {
    Config analysisConfig;
    BitField<decltype(analysisConfig.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    analysisConfig.eventType = eventBit.getValue();
    ClientId clientId = 0;
    MemOpRecord record1;
    record1.type = RecordType::MEMORY_RECORD;
    record1.recordIndex = 1;
    record1.space = MemOpSpace::INVALID;
    record1.subtype = RecordSubType::FREE;
    record1.addr = 0x7958;
    record1.memSize = 0;

    EXPECT_TRUE(HalAnalyzer::GetInstance(analysisConfig).Record(clientId, record1));
}

TEST(HalAnalyzerTest, do_memory_record_nulltable) {
    Config analysisConfig;
    BitField<decltype(analysisConfig.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    analysisConfig.eventType = eventBit.getValue();
    MemOpRecord record;
    record.type = RecordType::MEMORY_RECORD;
    record.recordIndex = 123;
    record.addr = 0x7958;
    ClientId clientId = 0;
    record.subtype = RecordSubType::FREE;
    EXPECT_TRUE(HalAnalyzer::GetInstance(analysisConfig).Record(clientId, record));
}