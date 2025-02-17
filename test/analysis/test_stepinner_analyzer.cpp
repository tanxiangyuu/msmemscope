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

MstxRecord CreatMstxRecord(MarkType type, const char* message, uint64_t stepId, uint64_t rangeId, uint64_t streamId)
{
    auto mstxRecord = MstxRecord {};
    mstxRecord.markType = type;
    strncpy_s(mstxRecord.markMessage, sizeof(mstxRecord.markMessage), message,
    sizeof(mstxRecord.markMessage));
    mstxRecord.stepId = stepId;
    mstxRecord.rangeId = rangeId;
    mstxRecord.streamId = streamId;

    return mstxRecord;
}

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
    memoryusage2.dataType = 2;
    memoryusage2.ptr = 12345;
    memoryusage2.allocSize = -512;
    memoryusage2.totalAllocated = 0;
    npuRecordFree.memoryUsage = memoryusage2;
    record2.record.torchNpuRecord = npuRecordFree;

    EXPECT_TRUE(stepinneranalyzer.Record(clientId, record1));
    EXPECT_TRUE(stepinneranalyzer.Record(clientId, record2));
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
    mstxRecordStart1.stepId = 1;
    mstxRecordStart1.streamId = 123;

    // 经过第二个step，但仍然未释放
    auto mstxRecordStart2 = MstxRecord {};
    mstxRecordStart2.markType = MarkType::RANGE_START_A;
    strncpy_s(mstxRecordStart2.markMessage, sizeof(mstxRecordStart2.markMessage), "step start",
    sizeof(mstxRecordStart2.markMessage));
    mstxRecordStart2.stepId = 2;
    mstxRecordStart2.streamId = 123;

    auto mstxRecordEnd = MstxRecord {};
    mstxRecordEnd.markType = MarkType::RANGE_END;
    mstxRecordEnd.stepId = 2;
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

    EXPECT_TRUE(analyzer->Record(clientId, record1));
    EXPECT_TRUE(MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordStart1));
    EXPECT_TRUE(MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordStart2));
    EXPECT_TRUE(MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordEnd));
}

TEST(StepInnerAnalyzerTest, do_npu_malloc_record_expect_sucess) {
    AnalysisConfig analysisConfig;
    StepInnerAnalyzer stepinneranalyzer{analysisConfig};
    ClientId clientId = 0;

    auto record = EventRecord{};
    record.type = RecordType::TORCH_NPU_RECORD;
    auto npuRecordMalloc = TorchNpuRecord {};
    npuRecordMalloc.recordIndex = 1;
    auto memoryusage = MemoryUsage {};
    memoryusage.deviceType = 20;
    memoryusage.deviceIndex = 0;
    memoryusage.dataType = 0;
    memoryusage.allocatorType = 0;
    memoryusage.ptr = 12345;
    memoryusage.allocSize = 512;
    memoryusage.totalAllocated = 512;
    memoryusage.totalActive = 512;
    memoryusage.totalReserved = 1024;
    memoryusage.streamPtr = 4321;
    npuRecordMalloc.memoryUsage = memoryusage;
    record.record.torchNpuRecord = npuRecordMalloc;

    EXPECT_TRUE(stepinneranalyzer.Record(clientId, record));
}

TEST(StepInnerAnalyzerTest, do_npu_malloc_record_expect_double_malloc) {
    AnalysisConfig analysisConfig;
    StepInnerAnalyzer stepinneranalyzer{analysisConfig};
    ClientId clientId = 0;

    auto record = EventRecord{};
    record.type = RecordType::TORCH_NPU_RECORD;
    auto npuRecordMalloc = TorchNpuRecord {};
    npuRecordMalloc.recordIndex = 1;
    auto memoryusage = MemoryUsage {};
    memoryusage.dataType = 0;
    memoryusage.ptr = 12345;
    memoryusage.allocSize = 512;
    npuRecordMalloc.memoryUsage = memoryusage;
    record.record.torchNpuRecord = npuRecordMalloc;

    // 地址重复的申请
    auto double_record = EventRecord{};
    double_record.type = RecordType::TORCH_NPU_RECORD;
    auto double_npuRecordMalloc = TorchNpuRecord {};
    double_npuRecordMalloc.recordIndex = 2;
    auto double_memoryusage = MemoryUsage {};
    double_memoryusage.dataType = 0;
    double_memoryusage.ptr = 12345;
    double_memoryusage.allocSize = 512;
    double_npuRecordMalloc.memoryUsage = double_memoryusage;
    double_record.record.torchNpuRecord = double_npuRecordMalloc;

    EXPECT_TRUE(stepinneranalyzer.Record(clientId, record));
    EXPECT_TRUE(stepinneranalyzer.Record(clientId, double_record));
}


TEST(StepInnerAnalyzerTest, do_npu_free_record_expect_free_error) {
    AnalysisConfig analysisConfig;
    StepInnerAnalyzer stepinneranalyzer{analysisConfig};
    ClientId clientId = 0;

    auto record = EventRecord{};
    record.type = RecordType::TORCH_NPU_RECORD;
    auto npuRecordFree = TorchNpuRecord {};
    npuRecordFree.recordIndex = 2;
    auto memoryusage = MemoryUsage {};
    memoryusage.deviceType = 20;
    memoryusage.deviceIndex = 0;
    memoryusage.dataType = 2;
    memoryusage.allocatorType = 0;
    memoryusage.ptr = 12345;
    memoryusage.allocSize = -512;
    memoryusage.totalAllocated = 0;
    memoryusage.totalActive = 0;
    memoryusage.totalReserved = 1024;
    memoryusage.streamPtr = 4321;
    npuRecordFree.memoryUsage = memoryusage;
    record.record.torchNpuRecord = npuRecordFree;

    EXPECT_TRUE(stepinneranalyzer.Record(clientId, record));
}

TEST(StepInnerAnalyzerTest, do_reveive_mstxmsg_expect_leaks) {
    AnalysisConfig analysisConfig;
    AnalyzerFactory analyzerfactory{analysisConfig};
    RecordType type = RecordType::TORCH_NPU_RECORD;
    std::shared_ptr<AnalyzerBase> analyzer = analyzerfactory.CreateAnalyzer(type);
    MstxAnalyzer::Instance().RegisterAnalyzer(analyzer);

    ClientId clientId = 0;
    // 第一个step会跳过
    auto mstxRecordStartFirst = CreatMstxRecord(MarkType::RANGE_START_A, "step start", 1, 1, 123);
    auto mstxRecordEndFirst = CreatMstxRecord(MarkType::RANGE_END, "", 1, 1, 123);


    // 第二个step发生泄漏
    auto mstxRecordStartSecond = CreatMstxRecord(MarkType::RANGE_START_A, "step start", 2, 2, 123);
    auto mstxRecordEndSecond = CreatMstxRecord(MarkType::RANGE_END, "", 2, 2, 123);


    // 构造第三个step时仍未释放的情景
    auto mstxRecordStartThird = CreatMstxRecord(MarkType::RANGE_START_A, "step start", 3, 3, 123);
    auto mstxRecordEndThird = CreatMstxRecord(MarkType::RANGE_END, "", 3, 3, 123);


    // step前后allocated内存不一致
    auto record = EventRecord{};
    record.type = RecordType::TORCH_NPU_RECORD;
    auto npuRecordMalloc = TorchNpuRecord {};
    npuRecordMalloc.recordIndex = 1;
    auto memoryusage = MemoryUsage {};
    memoryusage.dataType = 0;
    memoryusage.ptr = 12345;
    memoryusage.allocSize = 512;
    memoryusage.totalAllocated = 512;
    npuRecordMalloc.memoryUsage = memoryusage;
    record.record.torchNpuRecord = npuRecordMalloc;

    MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordStartFirst);
    MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordEndFirst);
    MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordStartSecond);
    EXPECT_TRUE(analyzer->Record(clientId, record));
    MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordStartSecond);
    EXPECT_TRUE(MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordStartThird));
    EXPECT_TRUE(MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordEndThird));
    analyzer->~AnalyzerBase();
}
