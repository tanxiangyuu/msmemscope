// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 
#include <gtest/gtest.h>
#include "command.h"
 
using namespace Leaks;
 
TEST(Command, run_ls_command_expect_success)
{
    std::vector<std::string> paramList = {"/bin/ls"};
    AnalysisConfig config;
    Command command(config);
    command.Exec(paramList);
}

TEST(Command, do_dump_record_except_success)
{
    DumpRecord dump{};
    ClientId clientId = 0;
    EventRecord record{};

    DumpHandler(clientId, dump, record);
}

TEST(Command, do_record_handler_except_success)
{
    MstxAnalyzer mstxanalyzer;
    AnalysisConfig analysisConfig;
    AnalyzerFactory analyzerfactory{analysisConfig};

    ClientId clientId = 0;
    auto record1 = EventRecord{};
    record1.type = RecordType::TORCH_NPU_RECORD;
    auto npuRecordMalloc = TorchNpuRecord {};
    npuRecordMalloc.recordIndex = 1;
    auto memoryusage1 = MemoryUsage {};
    memoryusage1.dataType = 0;
    memoryusage1.ptr = 12345;
    memoryusage1.allocSize = 512;
    memoryusage1.totalAllocated = 512;
    npuRecordMalloc.memoryUsage = memoryusage1;
    record1.record.torchNpuRecord = npuRecordMalloc;

    auto record2 = EventRecord{};
    record2.type = RecordType::MSTX_MARK_RECORD;
    auto mstxRecordStart = MstxRecord {};
    mstxRecordStart.markType = MarkType::RANGE_START_A;
    mstxRecordStart.rangeId = 1;
    mstxRecordStart.streamId = 123;
    record2.record.mstxRecord = mstxRecordStart;

    RecordHandler(clientId, record1, mstxanalyzer, analyzerfactory);
    RecordHandler(clientId, record2, mstxanalyzer, analyzerfactory);
}