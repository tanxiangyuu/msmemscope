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