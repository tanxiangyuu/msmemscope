// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <string>
#include "analyzer.h"
#include "record_info.h"
#include "config_info.h"

using namespace Leaks;

TEST(AnalyzerTest, AnalyzerConstruct) {
    AnalysisConfig analysisConfig;
    Analyzer analyzer(analysisConfig);
}


TEST(Analyzer, do_memory_record_expect_success)
{
    AnalysisConfig config;
    Analyzer analyzer(config);

    auto record = EventRecord{};
    record.type = RecordType::MEMORY_RECORD;
    record.flag = 2377900603261207558;
    auto memRecordMalloc = MemOpRecord {};
    memRecordMalloc.recordIndex = 123;
    memRecordMalloc.addr = 0x7958;
    memRecordMalloc.memSize = 1024;
    memRecordMalloc.timeStamp = 1234567;
    memRecordMalloc.memType = MemOpType::MALLOC;
    record.record.memoryRecord = memRecordMalloc;
    analyzer.Do(record);

    auto memRecordFree = memRecordMalloc;
    memRecordFree.memType = MemOpType::FREE;
    record.record.memoryRecord = memRecordFree;
    analyzer.Do(record);
}

TEST(Analyzer, do_kernellaunch_record_expect_success)
{
    AnalysisConfig config;
    Analyzer analyzer(config);

    auto record = EventRecord{};
    record.type = RecordType::KERNEL_LAUNCH_RECORD;
    analyzer.Do(record);
}

TEST(Analyzer, do_aclitf_record_expect_success)
{
    AnalysisConfig config;
    Analyzer analyzer(config);

    auto record = EventRecord{};
    record.type = RecordType::ACL_ITF_RECORD;
    analyzer.Do(record);

}

TEST(AnalyzerTest, AnalyzerLeakAnalyze) {
    AnalysisConfig config;
    Analyzer analyzer(config);

    auto record = EventRecord{};
    record.flag = 2377900603261207558;
    record.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc = MemOpRecord {};
    memRecordMalloc.recordIndex = 123;
    memRecordMalloc.addr = 0x7958;
    memRecordMalloc.memSize = 1024;
    memRecordMalloc.timeStamp = 1234567;
    memRecordMalloc.memType = MemOpType::MALLOC;
    record.record.memoryRecord = memRecordMalloc;

    analyzer.Do(record);
    analyzer.LeakAnalyze();
}

TEST(MemoryHashTableTest,BasicOperations) {
    MemoryHashTable memhashteble;

    auto record1 = EventRecord{};
    record1.flag = 2377900603261207558;
    record1.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc1 = MemOpRecord {};
    memRecordMalloc1.recordIndex = 1;
    memRecordMalloc1.space = MemOpSpace::DEVICE;
    memRecordMalloc1.memType = MemOpType::MALLOC;
    memRecordMalloc1.addr = 0x7958;
    memRecordMalloc1.memSize = 1024;
    memRecordMalloc1.timeStamp = 1234567;
    record1.record.memoryRecord = memRecordMalloc1;
    MemOpRecordKey memkey1(memRecordMalloc1.addr);

    auto record2 = EventRecord{};
    record2.flag = 504403158308635654;
    record2.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc2 = MemOpRecord {};
    memRecordMalloc2.recordIndex = 2;
    memRecordMalloc2.space = MemOpSpace::INVALID;
    memRecordMalloc2.memType = MemOpType::MALLOC;
    memRecordMalloc2.addr = 0x7957;
    memRecordMalloc2.memSize = 512;
    memRecordMalloc2.timeStamp = 1234568;
    record2.record.memoryRecord = memRecordMalloc2;
    MemOpRecordKey memkey2(memRecordMalloc2.addr);

    auto record3 = EventRecord{};
    record3.type = RecordType::MEMORY_RECORD;
    auto memRecordFree = MemOpRecord {};
    memRecordFree.recordIndex = 3;
    memRecordFree.space = MemOpSpace::INVALID;
    memRecordFree.memType = MemOpType::FREE;
    memRecordFree.addr = 0x7958;
    memRecordFree.memSize = 0;
    record3.record.memoryRecord = memRecordFree;

    auto record4 = EventRecord{};
    record4.flag = 504403158275081222;
    record4.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc4 = MemOpRecord {};
    memRecordMalloc4.recordIndex = 4;
    memRecordMalloc4.space = MemOpSpace::DEVICE;
    memRecordMalloc4.memType = MemOpType::MALLOC;
    memRecordMalloc4.addr = 0x7960;
    memRecordMalloc4.memSize = 1024;
    memRecordMalloc4.timeStamp = 1234557;
    record4.record.memoryRecord = memRecordMalloc4;
    MemOpRecordKey memkey4(memRecordMalloc4.addr);

    memhashteble.Record(record1);
    memhashteble.Record(record2);
    memhashteble.Record(record3);
    memhashteble.Record(record4);
    memhashteble.CheckLeak();
}