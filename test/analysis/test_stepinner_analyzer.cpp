// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <string>
#include "securec.h"
#include "stepinner_analyzer.h"
#include "mstx_analyzer.h"
#include "analyzer_factory.h"
#include "record_info.h"
#include "config_info.h"

using namespace Leaks;

TEST(StepInnerAnalyzerTest, do_npu_free_record_expect_sucess) {
    AnalysisConfig analysisConfig;
    StepInnerAnalyzer stepinneranalyzer{analysisConfig};
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
    record2.type = RecordType::TORCH_NPU_RECORD;
    auto npuRecordFree = TorchNpuRecord {};
    npuRecordFree.recordIndex = 2;
    auto memoryusage2 = MemoryUsage {};
    memoryusage2.dataType = 1;
    memoryusage2.ptr = 12345;
    memoryusage2.allocSize = -512;
    memoryusage2.totalAllocated = 0;
    npuRecordFree.memoryUsage = memoryusage2;
    record2.record.torchNpuRecord = npuRecordFree;

    stepinneranalyzer.Record(clientId, record1);
    stepinneranalyzer.Record(clientId, record2);
}

TEST(StepInnerAnalyzerTest, do_reveive_mstxmsg_expect_leaks_warning) {
    AnalysisConfig analysisConfig;
    AnalyzerFactory analyzerfactory{analysisConfig};
    RecordType type = RecordType::TORCH_NPU_RECORD;
    std::shared_ptr<AnalyzerBase> analyzer = analyzerfactory.CreateAnalyzer(type);
    MstxAnalyzer::Instance().RegisterAnalyzer(analyzer);

    ClientId clientId = 0;
    auto mstxRecordStart1 = MstxRecord {};
    mstxRecordStart1.markType = MarkType::RANGE_START_A;
    strncpy_s(mstxRecordStart1.markMessage, sizeof(mstxRecordStart1.markMessage), "step start",
    sizeof(mstxRecordStart1.markMessage));
    mstxRecordStart1.rangeId = 1;
    mstxRecordStart1.streamId = 123;

    // 经过第二个step，但仍然未释放
    auto mstxRecordStart2 = MstxRecord {};
    mstxRecordStart2.markType = MarkType::RANGE_START_A;
    strncpy_s(mstxRecordStart2.markMessage, sizeof(mstxRecordStart2.markMessage), "step start",
    sizeof(mstxRecordStart2.markMessage));
    mstxRecordStart2.rangeId = 2;
    mstxRecordStart2.streamId = 123;

    auto mstxRecordEnd = MstxRecord {};
    mstxRecordEnd.markType = MarkType::RANGE_END;
    mstxRecordEnd.rangeId = 2;
    mstxRecordEnd.streamId = 123;

    // 经过两个step的内存
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

    analyzer->Record(clientId, record1);
    MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordStart1);
    MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordStart2);
    MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordEnd);
}