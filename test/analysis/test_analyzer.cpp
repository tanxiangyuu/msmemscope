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