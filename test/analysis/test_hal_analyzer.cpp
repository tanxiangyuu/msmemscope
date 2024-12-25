// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <string>
#include "hal_analyzer.h"
#include "record_info.h"
#include "config_info.h"

using namespace Leaks;

TEST(HalAnalyzerTest, do_hal_record_except_leaks) {
    AnalysisConfig analysisConfig;
    HalAnalyzer halanalyzer{analysisConfig};
 
    ClientId clientId = 0;
    auto record1 = EventRecord{};
    record1.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc1 = MemOpRecord {};
    memRecordMalloc1.flag = 2377900603261207558;
    memRecordMalloc1.recordIndex = 1;
    memRecordMalloc1.space = MemOpSpace::DEVICE;
    memRecordMalloc1.memType = MemOpType::MALLOC;
    memRecordMalloc1.addr = 0x7958;
    memRecordMalloc1.memSize = 1024;
    memRecordMalloc1.timeStamp = 1234567;
    record1.record.memoryRecord = memRecordMalloc1;
 
    auto record2 = EventRecord{};
    record2.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc2 = MemOpRecord {};
    memRecordMalloc2.flag = 18374686480754951175;
    memRecordMalloc2.recordIndex = 2;
    memRecordMalloc2.space = MemOpSpace::INVALID;
    memRecordMalloc2.memType = MemOpType::MALLOC;
    memRecordMalloc2.addr = 0x7957;
    memRecordMalloc2.memSize = 512;
    memRecordMalloc2.timeStamp = 1234568;
    record2.record.memoryRecord = memRecordMalloc2;
 
    auto record4 = EventRecord{};
    record4.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc4 = MemOpRecord {};
    memRecordMalloc4.flag = 504403158275081222;
    memRecordMalloc4.recordIndex = 4;
    memRecordMalloc4.space = MemOpSpace::DEVICE;
    memRecordMalloc4.memType = MemOpType::MALLOC;
    memRecordMalloc4.addr = 0x7960;
    memRecordMalloc4.memSize = 1024;
    memRecordMalloc4.timeStamp = 1234557;
    record4.record.memoryRecord = memRecordMalloc4;
 
    halanalyzer.Record(clientId, record1);
    halanalyzer.Record(clientId, record2);
    halanalyzer.Record(clientId, record4);
}

TEST(HalAnalyzerTest, do_record_except_no_leaks) {
    AnalysisConfig analysisConfig;
    HalAnalyzer halanalyzer{analysisConfig};

    ClientId clientId = 0;
    auto record1 = EventRecord{};
    record1.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc1 = MemOpRecord {};
    memRecordMalloc1.flag = 2377900603261207558;
    memRecordMalloc1.recordIndex = 1;
    memRecordMalloc1.space = MemOpSpace::DEVICE;
    memRecordMalloc1.memType = MemOpType::MALLOC;
    memRecordMalloc1.addr = 0x7958;
    memRecordMalloc1.memSize = 1024;
    memRecordMalloc1.timeStamp = 1234567;
    record1.record.memoryRecord = memRecordMalloc1;

    auto record3 = EventRecord{};
    record3.type = RecordType::MEMORY_RECORD;
    auto memRecordFree = MemOpRecord {};
    memRecordFree.recordIndex = 3;
    memRecordFree.space = MemOpSpace::INVALID;
    memRecordFree.memType = MemOpType::FREE;
    memRecordFree.addr = 0x7958;
    memRecordFree.memSize = 0;
    record3.record.memoryRecord = memRecordFree;

    halanalyzer.Record(clientId, record1);
    halanalyzer.Record(clientId, record3);
}

TEST(HalAnalyzerTest, do_record_excpet_double_free) {
    AnalysisConfig analysisConfig;
    HalAnalyzer halanalyzer{analysisConfig};

    ClientId clientId = 0;
    auto record1 = EventRecord{};
    record1.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc1 = MemOpRecord {};
    memRecordMalloc1.flag = 2377900603261207558;
    memRecordMalloc1.recordIndex = 1;
    memRecordMalloc1.space = MemOpSpace::DEVICE;
    memRecordMalloc1.memType = MemOpType::MALLOC;
    memRecordMalloc1.addr = 0x7958;
    memRecordMalloc1.memSize = 1024;
    memRecordMalloc1.timeStamp = 1234567;
    record1.record.memoryRecord = memRecordMalloc1;

    auto record2 = EventRecord{};
    record2.type = RecordType::MEMORY_RECORD;
    auto memRecordFree1 = MemOpRecord {};
    memRecordFree1.recordIndex = 2;
    memRecordFree1.space = MemOpSpace::INVALID;
    memRecordFree1.memType = MemOpType::FREE;
    memRecordFree1.addr = 0x7958;
    memRecordFree1.memSize = 0;
    record2.record.memoryRecord = memRecordFree1;

    auto record3 = EventRecord{};
    record3.type = RecordType::MEMORY_RECORD;
    auto memRecordFree2 = MemOpRecord {};
    memRecordFree2.recordIndex = 3;
    memRecordFree2.space = MemOpSpace::INVALID;
    memRecordFree2.memType = MemOpType::FREE;
    memRecordFree2.addr = 0x7958;
    memRecordFree2.memSize = 0;
    record3.record.memoryRecord = memRecordFree2;

    halanalyzer.Record(clientId, record1);
    halanalyzer.Record(clientId, record2);
    halanalyzer.Record(clientId, record3);
}

TEST(HalAnalyzerTest, do_record_except_double_malloc) {
    AnalysisConfig analysisConfig;
    HalAnalyzer halanalyzer{analysisConfig};

    ClientId clientId = 0;
    auto record1 = EventRecord{};
    record1.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc1 = MemOpRecord {};
    memRecordMalloc1.flag = 2377900603261207558;
    memRecordMalloc1.recordIndex = 1;
    memRecordMalloc1.space = MemOpSpace::DEVICE;
    memRecordMalloc1.memType = MemOpType::MALLOC;
    memRecordMalloc1.addr = 0x7958;
    memRecordMalloc1.memSize = 1024;
    memRecordMalloc1.timeStamp = 1234567;
    record1.record.memoryRecord = memRecordMalloc1;

    auto record2 = EventRecord{};
    record2.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc2 = MemOpRecord {};
    memRecordMalloc2.flag = 2377900603261207558;
    memRecordMalloc2.recordIndex = 2;
    memRecordMalloc2.space = MemOpSpace::DEVICE;
    memRecordMalloc2.memType = MemOpType::MALLOC;
    memRecordMalloc2.addr = 0x7958;
    memRecordMalloc2.memSize = 1024;
    memRecordMalloc2.timeStamp = 1234567;
    record2.record.memoryRecord = memRecordMalloc2;

    halanalyzer.Record(clientId, record1);
    halanalyzer.Record(clientId, record2);
}

TEST(HalAnalyzerTest, do_record_except_free_null) {
    AnalysisConfig analysisConfig;
    HalAnalyzer halanalyzer{analysisConfig};

    ClientId clientId = 0;
    auto record1 = EventRecord{};
    record1.type = RecordType::MEMORY_RECORD;
    auto memRecordFree1 = MemOpRecord {};
    memRecordFree1.recordIndex = 1;
    memRecordFree1.space = MemOpSpace::INVALID;
    memRecordFree1.memType = MemOpType::FREE;
    memRecordFree1.addr = 0x7958;
    memRecordFree1.memSize = 0;
    record1.record.memoryRecord = memRecordFree1;

    halanalyzer.Record(clientId, record1);
}

TEST(HalAnalyzerTest, do_record_fail) {
    AnalysisConfig analysisConfig;
    HalAnalyzer halanalyzer{analysisConfig};

    ClientId clientId = 0;
    auto record1 = EventRecord{};
    record1.type = RecordType::MEMORY_RECORD;
    auto memRecordFree1 = MemOpRecord {};
    memRecordFree1.recordIndex = 1;
    memRecordFree1.space = MemOpSpace::INVALID;
    memRecordFree1.memType = MemOpType::FREE;
    memRecordFree1.addr = 0x7958;
    memRecordFree1.memSize = 0;
    record1.record.memoryRecord = memRecordFree1;

    halanalyzer.Record(clientId, record1);
}

TEST(HalAnalyzerTest, do_memory_record_nulltable) {
    AnalysisConfig analysisConfig;
    HalAnalyzer halanalyzer{analysisConfig};

    auto record = EventRecord{};
    record.type = RecordType::MEMORY_RECORD;
    auto memRecordFree = MemOpRecord {};
    memRecordFree.recordIndex = 123;
    memRecordFree.addr = 0x7958;
    ClientId clientId = 0;
    memRecordFree.memType = MemOpType::FREE;
    record.record.memoryRecord = memRecordFree;
    halanalyzer.Record(clientId, record);
}