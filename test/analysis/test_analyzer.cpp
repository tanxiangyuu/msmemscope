// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 
#include "analysis/analyzer.h"
 
#include <gtest/gtest.h>
 
using namespace Leaks;
 
TEST(AnalyzerTest, AnalyzerConstruct) {
    AnalysisConfig analysisConfig;
    Analyzer analyzer(analysisConfig);
}
 
TEST(AnalyzerTest, AnalyzerDoMalloc) {
    EventRecord eventRecord;
    eventRecord.record.memoryRecord.memType = MemOpType::MALLOC;
    AnalysisConfig analysisConfig;
    Analyzer analyzer(analysisConfig);
    analyzer.Do(eventRecord);
}
 
TEST(AnalyzerTest, AnalyzerDoFree) {
    EventRecord eventRecord;
    eventRecord.record.memoryRecord.memType = MemOpType::FREE;
    AnalysisConfig analysisConfig;
    Analyzer analyzer(analysisConfig);
    analyzer.Do(eventRecord);
}

TEST(AnalyzerTest, AnalyzerLeakAnalyze) {
    EventRecord eventRecord;
    eventRecord.record.memoryRecord.memType = MemOpType::MALLOC;
    AnalysisConfig analysisConfig;
    Analyzer analyzer(analysisConfig);
    analyzer.Do(eventRecord);
    analyzer.LeakAnalyze();
}

TEST(MemoryHashTableTest,BasicOperations) {
    MemoryHashTable memhashteble;

    MemOpRecord record1;
    record1.recordIndex = 1;
    record1.space = MemOpSpace::DEVICE;
    record1.memType = MemOpType::MALLOC;
    MemOpRecord record2;
    record2.recordIndex = 1;
    record2.space = MemOpSpace::INVALID;
    record2.memType = MemOpType::MALLOC;
    MemOpRecord record3;
    record3.recordIndex = 1;
    record3.space = MemOpSpace::DEVICE;
    record3.memType = MemOpType::FREE;
    MemOpRecord record4;
    record4.recordIndex = 2;
    record4.space = MemOpSpace::HOST;
    record4.memType = MemOpType::MALLOC;

    memhashteble.Record(record1);
    EXPECT_TRUE(memhashteble.table.find(record1) != memhashteble.table.end());
    memhashteble.Record(record2);
    EXPECT_TRUE(memhashteble.table.find(record2) == memhashteble.table.end());
    memhashteble.Record(record3);
    EXPECT_TRUE(memhashteble.table.find(record3) == memhashteble.table.end());
    memhashteble.Record(record4);
    EXPECT_TRUE(memhashteble.table.find(record4) != memhashteble.table.end());

    memhashteble.CheckLeak();
}