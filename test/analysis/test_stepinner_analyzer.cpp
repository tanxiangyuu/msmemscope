// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <string>
#include "securec.h"
#define private public
#include "stepinner_analyzer.h"
#undef private
#include "bit_field.h"
#include "mstx_analyzer.h"
#include "record_info.h"
#include "config_info.h"

using namespace Leaks;

MstxRecord CreatMstxRecord(MarkType type, const char* message, uint64_t stepId, uint64_t rangeId, uint64_t streamId)
{
    auto mstxRecord = MstxRecord {};
    mstxRecord.markType = type;
    strncpy_s(mstxRecord.markMessage, sizeof(mstxRecord.markMessage), message,
    sizeof(mstxRecord.markMessage) - 1);
    mstxRecord.markMessage[sizeof(mstxRecord.markMessage) - 1] = '\0';
    mstxRecord.devId = 0;
    mstxRecord.stepId = stepId;
    mstxRecord.rangeId = rangeId;
    mstxRecord.streamId = streamId;

    return mstxRecord;
}

TEST(StepInnerAnalyzerTest, do_npu_free_record_expect_sucess) {
    // 先初始化注册
    Config analysisConfig;
    BitField<decltype(analysisConfig.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    analysisConfig.eventType = eventBit.getValue();
    analysisConfig.stepList.stepCount = 0;
    StepInnerAnalyzer::GetInstance(analysisConfig).config_.stepList.stepCount = 0;
    static StepInnerAnalyzer analyzer(analysisConfig);
    ClientId clientId = 0;

    auto record1 = EventRecord{};
    record1.type = RecordType::TORCH_NPU_RECORD;
    auto npuRecordMalloc = TorchNpuRecord {};
    npuRecordMalloc.recordIndex = 1;
    auto memoryusage1 = MemoryUsage {};
    memoryusage1.deviceIndex = 0;
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
    memoryusage2.deviceIndex = 0;
    memoryusage2.dataType = 2;
    memoryusage2.ptr = 12345;
    memoryusage2.allocSize = -512;
    memoryusage2.totalAllocated = 0;
    npuRecordFree.memoryUsage = memoryusage2;
    record2.record.torchNpuRecord = npuRecordFree;

    EXPECT_TRUE(StepInnerAnalyzer::GetInstance(analysisConfig).Record(clientId, record1));
    EXPECT_TRUE(StepInnerAnalyzer::GetInstance(analysisConfig).Record(clientId, record2));
}

TEST(StepInnerAnalyzerTest, do_reveive_mstxmsg_expect_leaks_warning)
{
    ClientId clientId = 0;
    auto mstxRecordStart1 = MstxRecord {};
    mstxRecordStart1.markType = MarkType::RANGE_START_A;
    strncpy_s(mstxRecordStart1.markMessage, sizeof(mstxRecordStart1.markMessage), "step start",
    sizeof(mstxRecordStart1.markMessage));
    mstxRecordStart1.devId = 0;
    mstxRecordStart1.stepId = 1;
    mstxRecordStart1.streamId = 123;

    // 经过第二个step，但仍然未释放
    auto mstxRecordStart2 = MstxRecord {};
    mstxRecordStart2.markType = MarkType::RANGE_START_A;
    strncpy_s(mstxRecordStart2.markMessage, sizeof(mstxRecordStart2.markMessage), "step start",
    sizeof(mstxRecordStart2.markMessage));
    mstxRecordStart2.devId = 0;
    mstxRecordStart2.stepId = 2;
    mstxRecordStart2.streamId = 123;

    auto mstxRecordEnd = MstxRecord {};
    mstxRecordEnd.markType = MarkType::RANGE_END;
    mstxRecordEnd.devId = 0;
    mstxRecordEnd.stepId = 2;
    mstxRecordEnd.streamId = 123;

    // 经过两个step的内存
    auto record1 = EventRecord{};
    record1.type = RecordType::TORCH_NPU_RECORD;
    auto npuRecordMalloc = TorchNpuRecord {};
    npuRecordMalloc.recordIndex = 1;
    auto memoryusage1 = MemoryUsage {};
    memoryusage1.deviceIndex = 0;
    memoryusage1.dataType = 0;
    memoryusage1.ptr = 12345;
    memoryusage1.allocSize = 512;
    memoryusage1.totalAllocated = 512;
    npuRecordMalloc.memoryUsage = memoryusage1;
    record1.record.torchNpuRecord = npuRecordMalloc;

    // 先初始化注册
    Config analysisConfig;
    BitField<decltype(analysisConfig.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    analysisConfig.eventType = eventBit.getValue();
    analysisConfig.stepList.stepCount = 0;
    StepInnerAnalyzer::GetInstance(analysisConfig).config_.stepList.stepCount = 0;
    static StepInnerAnalyzer analyzer(analysisConfig);
    EXPECT_TRUE(StepInnerAnalyzer::GetInstance(analysisConfig).Record(clientId, record1));
    EXPECT_TRUE(MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordStart1));
    EXPECT_TRUE(MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordStart2));
    EXPECT_TRUE(MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordEnd));
}

TEST(StepInnerAnalyzerTest, do_npu_malloc_record_expect_sucess) {
    // 先初始化注册
    Config analysisConfig;
    BitField<decltype(analysisConfig.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    analysisConfig.eventType = eventBit.getValue();
    analysisConfig.stepList.stepCount = 0;
    StepInnerAnalyzer::GetInstance(analysisConfig).config_.stepList.stepCount = 0;
    static StepInnerAnalyzer analyzer(analysisConfig);
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

    EXPECT_TRUE(StepInnerAnalyzer::GetInstance(analysisConfig).Record(clientId, record));
}

TEST(StepInnerAnalyzerTest, do_npu_malloc_record_expect_double_malloc) {
    // 先初始化注册
    Config analysisConfig;
    BitField<decltype(analysisConfig.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    analysisConfig.eventType = eventBit.getValue();
    analysisConfig.stepList.stepCount = 0;
    StepInnerAnalyzer::GetInstance(analysisConfig).config_.stepList.stepCount = 0;
    static StepInnerAnalyzer analyzer(analysisConfig);
    ClientId clientId = 0;

    auto record = EventRecord{};
    record.type = RecordType::TORCH_NPU_RECORD;
    auto npuRecordMalloc = TorchNpuRecord {};
    npuRecordMalloc.recordIndex = 1;
    auto memoryusage = MemoryUsage {};
    memoryusage.deviceIndex = 0;
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
    double_memoryusage.deviceIndex = 0;
    double_memoryusage.dataType = 0;
    double_memoryusage.ptr = 12345;
    double_memoryusage.allocSize = 512;
    double_npuRecordMalloc.memoryUsage = double_memoryusage;
    double_record.record.torchNpuRecord = double_npuRecordMalloc;

    EXPECT_TRUE(StepInnerAnalyzer::GetInstance(analysisConfig).Record(clientId, record));
    EXPECT_TRUE(StepInnerAnalyzer::GetInstance(analysisConfig).Record(clientId, double_record));
}


TEST(StepInnerAnalyzerTest, do_npu_free_record_expect_free_error) {
    // 先初始化注册
    Config analysisConfig;
    BitField<decltype(analysisConfig.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    analysisConfig.eventType = eventBit.getValue();
    analysisConfig.stepList.stepCount = 0;
    StepInnerAnalyzer::GetInstance(analysisConfig).config_.stepList.stepCount = 0;
    static StepInnerAnalyzer analyzer(analysisConfig);
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

    EXPECT_TRUE(StepInnerAnalyzer::GetInstance(analysisConfig).Record(clientId, record));
}

TEST(StepInnerAnalyzerTest, do_reveive_mstxmsg_expect_leaks) {
    ClientId clientId = 0;
    Leaks::DeviceId deviceId = 0;
    // 先初始化注册
    Config analysisConfig;
    BitField<decltype(analysisConfig.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    analysisConfig.eventType = eventBit.getValue();
    analysisConfig.stepList.stepCount = 0;
    StepInnerAnalyzer::GetInstance(analysisConfig).config_.stepList.stepCount = 0;
    static StepInnerAnalyzer analyzer(analysisConfig);

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
    memoryusage.deviceIndex = 0;
    memoryusage.dataType = 0;
    memoryusage.ptr = 12345;
    memoryusage.allocSize = 512;
    memoryusage.totalAllocated = 512;
    npuRecordMalloc.memoryUsage = memoryusage;
    record.record.torchNpuRecord = npuRecordMalloc;

    MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordStartFirst);
    MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordEndFirst);
    MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordStartSecond);
    EXPECT_TRUE(StepInnerAnalyzer::GetInstance(analysisConfig).Record(clientId, record));
    MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordStartSecond);
    EXPECT_TRUE(MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordStartThird));
    EXPECT_TRUE(MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordEndThird));
    StepInnerAnalyzer::GetInstance(analysisConfig).ReportGap(deviceId);
    StepInnerAnalyzer::GetInstance(analysisConfig).ReportLeak(deviceId);
}

TEST(StepInnerAnalyzerTest, do_input_exist_deviceid_CreateTables_return_true)
{
    Config config;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.eventType = eventBit.getValue();
    StepInnerAnalyzer stepInner{config};
    NpuMemUsage npumemusage{};
    stepInner.npuMemUsages_.insert({1, npumemusage});
    auto ret = stepInner.CreateTables(1);
    ASSERT_TRUE(ret);
}

TEST(StepInnerAnalyzerTest, do_input_not_exist_deviceid_CreateTables_return_true)
{
    Config config;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.eventType = eventBit.getValue();
    StepInnerAnalyzer stepInner{config};
    auto ret = stepInner.CreateTables(1);
    ASSERT_TRUE(ret);
}

TEST(StepInnerAnalyzerTest, do_input_exist_deviceid_CreateMstxTables_return_true)
{
    Config config;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.eventType = eventBit.getValue();
    StepInnerAnalyzer stepInner{config};
    MstxRecordTable mstxrecordtable{};
    stepInner.mstxTables_.insert({1, mstxrecordtable});
    auto ret = stepInner.CreateMstxTables(1);
    ASSERT_TRUE(ret);
}

TEST(StepInnerAnalyzerTest, do_input_not_exist_deviceid_CreateMstxTables_return_true)
{
    Config config;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.eventType = eventBit.getValue();
    StepInnerAnalyzer stepInner{config};
    auto ret = stepInner.CreateMstxTables(1);
    ASSERT_TRUE(ret);
}

TEST(StepInnerAnalyzerTest, do_input_exist_deviceid_CreateLeakSumTables_return_true)
{
    Config config;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.eventType = eventBit.getValue();
    StepInnerAnalyzer stepInner{config};
    LeakSumsTable leaksumstable{};
    stepInner.leakMemSums_.insert({1, leaksumstable});
    auto ret = stepInner.CreateLeakSumTables(1);
    ASSERT_TRUE(ret);
}

TEST(StepInnerAnalyzerTest, do_input_not_exist_deviceid_CreateLeakSumTables_return_true)
{
    Config config;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.eventType = eventBit.getValue();
    StepInnerAnalyzer stepInner{config};
    auto ret = stepInner.CreateLeakSumTables(1);
    ASSERT_TRUE(ret);
}

TEST(StepInnerAnalyzerTest, do_input_steps_command_disable_analysis)
{
    Config config;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.eventType = eventBit.getValue();
    config.stepList.stepCount = 2;
    StepInnerAnalyzer stepInner{config};
    auto ret = stepInner.IsStepInnerAnalysisEnable();
    ASSERT_FALSE(ret);
}

TEST(StepInnerAnalyzerTest, do_not_input_steps_command_enable_analysis)
{
    Config config;
    config.stepList.stepCount = 0;
    BitField<decltype(config.eventType)> eventBit;
    BitField<decltype(config.analysisType)> analysisBit;
    analysisBit.setBit(static_cast<size_t>(AnalysisType::LEAKS_ANALYSIS));
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.analysisType = analysisBit.getValue();
    config.eventType = eventBit.getValue();
    StepInnerAnalyzer stepInner{config};
    auto ret = stepInner.IsStepInnerAnalysisEnable();
    ASSERT_TRUE(ret);
}

TEST(StepInnerAnalyzerTest, do_stepId_below_1_SkipCheck_return_true)
{
    Config config;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.eventType = eventBit.getValue();
    StepInnerAnalyzer stepInner{config};
    NpuMemInfo npuMemInfo{};
    npuMemInfo.stepId = 0;
    auto ret = stepInner.SkipCheck(npuMemInfo);
    ASSERT_TRUE(ret);
}

TEST(StepInnerAnalyzerTest, do_stepId_up_1_SkipCheck_return_true)
{
    Config config;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.eventType = eventBit.getValue();
    StepInnerAnalyzer stepInner{config};
    NpuMemInfo npuMemInfo{};
    npuMemInfo.stepId = 3;
    auto ret = stepInner.SkipCheck(npuMemInfo);
    ASSERT_FALSE(ret);
}

TEST(StepInnerAnalyzerTest, do_updateallocated_step_0_update_0)
{
    Config config;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.eventType = eventBit.getValue();
    config.stepList.stepCount = 2;
    StepInnerAnalyzer stepInner{config};
    NpuMemUsage npumemusage;
    npumemusage.mstxStep = 0;
    npumemusage.stepMaxAllocated = 0;
    npumemusage.stepMinAllocated = 0;
    stepInner.npuMemUsages_.insert({0, npumemusage});
    stepInner.UpdateAllocated(0, 100);
    ASSERT_EQ(stepInner.npuMemUsages_[0].stepMaxAllocated, 0);
    ASSERT_EQ(stepInner.npuMemUsages_[0].stepMinAllocated, 0);
}

TEST(StepInnerAnalyzerTest, do_updateallocated_step_2_update_allocated)
{
    Config config;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.eventType = eventBit.getValue();
    config.stepList.stepCount = 2;
    StepInnerAnalyzer stepInner{config};
    NpuMemUsage npumemusage;
    npumemusage.mstxStep = 2;

    npumemusage.stepMaxAllocated = 20;
    npumemusage.stepMinAllocated = 20;
    stepInner.npuMemUsages_.insert({0, npumemusage});
    stepInner.UpdateAllocated(0, 100);
    ASSERT_EQ(stepInner.npuMemUsages_[0].stepMaxAllocated, 20);
    ASSERT_EQ(stepInner.npuMemUsages_[0].stepMinAllocated, 20);
}

TEST(StepInnerAnalyzerTest, do_checkgap_minmaxallocratio_equal_0_expect_reset_allocated)
{
    Config config;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.eventType = eventBit.getValue();
    config.stepList.stepCount = 2;
    StepInnerAnalyzer stepInner{config};
    NpuMemUsage npumemusage;
    npumemusage.mstxStep = 2;
    npumemusage.stepMaxAllocated = 100;
    npumemusage.stepMinAllocated = 20;
    stepInner.npuMemUsages_.insert({0, npumemusage});
    stepInner.CheckGap(0);
    ASSERT_EQ(stepInner.npuMemUsages_[0].stepMaxAllocated, 0);
    ASSERT_EQ(stepInner.npuMemUsages_[0].stepMinAllocated, 0);
}

TEST(StepInnerAnalyzerTest, do_checkgap_minmaxallocratio_expect_true_allocated)
{
    Config config;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.eventType = eventBit.getValue();
    config.stepList.stepCount = 2;
    StepInnerAnalyzer stepInner{config};
    NpuMemUsage npumemusage;
    GapInfo gapinfo;
    gapinfo.minMaxAllocRatio = 0.1;
    npumemusage.mstxStep = 2;
    npumemusage.stepMaxAllocated = 100;
    npumemusage.stepMinAllocated = 20;
    npumemusage.maxGapInfo = gapinfo;
    stepInner.npuMemUsages_.insert({0, npumemusage});
    stepInner.CheckGap(0);
    ASSERT_EQ(stepInner.npuMemUsages_[0].stepMaxAllocated, 0);
    ASSERT_EQ(stepInner.npuMemUsages_[0].stepMinAllocated, 0);
}

TEST(StepInnerAnalyzerRecordFuncTest, Recordtest)
{
    Leaks::Config config;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.eventType = eventBit.getValue();
    config.enableCompare = false;
    config.inputCorrectPaths = true;
    config.outputCorrectPaths = false;
    config.stepList.stepCount = 0;
    ClientId clientId = 1;
    Leaks::EventRecord record;
    EXPECT_TRUE(Leaks::StepInnerAnalyzer::GetInstance(config).Record(clientId, record));
}

TEST(StepInnerAnalyzerRecordFuncTest, recordMallocSuccess) {
    // 先初始化注册
    Leaks::Config config;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.eventType = eventBit.getValue();
    config.enableCompare = false;
    config.inputCorrectPaths = true;
    config.outputCorrectPaths = false;
    config.stepList.stepCount = 0;
    ClientId clientId = 1;

    auto record1 = EventRecord{};
    record1.type = RecordType::TORCH_NPU_RECORD;
    auto npuRecordMalloc = TorchNpuRecord {};
    npuRecordMalloc.recordIndex = 1;
    auto memoryusage1 = MemoryUsage {};
    memoryusage1.deviceIndex = 0;
    memoryusage1.dataType = 0;
    memoryusage1.ptr = 12345;
    memoryusage1.allocSize = 512;
    memoryusage1.totalAllocated = 512;
    npuRecordMalloc.memoryUsage = memoryusage1;
    record1.record.torchNpuRecord = npuRecordMalloc;

    EXPECT_TRUE(StepInnerAnalyzer::GetInstance(config).Record(clientId, record1));
}

TEST(StepInnerAnalyzerRecordFuncTest, recordFreeSuccess) {
    // 先初始化注册
    Leaks::Config config;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.eventType = eventBit.getValue();
    config.enableCompare = false;
    config.inputCorrectPaths = true;
    config.outputCorrectPaths = false;
    config.stepList.stepCount = 0;
    ClientId clientId = 1;

    auto record1 = EventRecord{};
    record1.type = RecordType::TORCH_NPU_RECORD;
    auto npuRecordFree = TorchNpuRecord {};
    npuRecordFree.recordIndex = 1;
    auto memoryusage1 = MemoryUsage {};
    memoryusage1.deviceIndex = 0;
    memoryusage1.dataType = 1;
    memoryusage1.ptr = 12345;
    memoryusage1.allocSize = 512;
    memoryusage1.totalAllocated = 512;
    npuRecordFree.memoryUsage = memoryusage1;
    record1.record.torchNpuRecord = npuRecordFree;

    EXPECT_TRUE(StepInnerAnalyzer::GetInstance(config).Record(clientId, record1));
}

TEST(StepInnerAnalyzerReceiveMstxMsgFuncTest, ReceiveMstxMsgIfRangeStartA) {
    // 先初始化注册
    Leaks::Config config;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.eventType = eventBit.getValue();
    config.enableCompare = false;
    config.inputCorrectPaths = true;
    config.outputCorrectPaths = false;
    config.stepList.stepCount = 0;
    ClientId clientId = 1;

    auto mstxRecordStart1 = MstxRecord {};
    mstxRecordStart1.markType = MarkType::RANGE_START_A;
    strncpy_s(mstxRecordStart1.markMessage, sizeof(mstxRecordStart1.markMessage), "step start",
    sizeof(mstxRecordStart1.markMessage));
    mstxRecordStart1.devId = 0;
    mstxRecordStart1.stepId = 1;
    mstxRecordStart1.streamId = 123;

    StepInnerAnalyzer::GetInstance(config).ReceiveMstxMsg(mstxRecordStart1);
}

TEST(StepInnerAnalyzerReceiveMstxMsgFuncTest, ReceiveMstxMsgIfRangeEnd) {
    // 先初始化注册
    Leaks::Config config;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.eventType = eventBit.getValue();
    config.enableCompare = false;
    config.inputCorrectPaths = true;
    config.outputCorrectPaths = false;
    config.stepList.stepCount = 0;
    ClientId clientId = 1;

    auto mstxRecordStart1 = MstxRecord {};
    mstxRecordStart1.markType = Leaks::MarkType::RANGE_END;
    strncpy_s(mstxRecordStart1.markMessage, sizeof(mstxRecordStart1.markMessage), "step end",
    sizeof(mstxRecordStart1.markMessage));
    mstxRecordStart1.devId = 0;
    mstxRecordStart1.stepId = 1;
    mstxRecordStart1.streamId = 123;

    StepInnerAnalyzer::GetInstance(config).ReceiveMstxMsg(mstxRecordStart1);
}

TEST(StepInnerAnalyzerUpdateAllocatedFuncTest, UpdateAllocatedUpdateMaxTest)
{
    Config config;
    config.stepList.stepCount = 0;
    BitField<decltype(config.eventType)> eventBit;
    BitField<decltype(config.analysisType)> analysisBit;
    analysisBit.setBit(static_cast<size_t>(AnalysisType::LEAKS_ANALYSIS));
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.analysisType = analysisBit.getValue();
    config.eventType = eventBit.getValue();
    StepInnerAnalyzer stepInner{config};
    NpuMemUsage npumemusage;
    npumemusage.mstxStep = 2;

    npumemusage.stepMaxAllocated = 20;
    npumemusage.stepMinAllocated = 20;
    stepInner.npuMemUsages_.insert({0, npumemusage});
    stepInner.UpdateAllocated(0, 100);
    ASSERT_EQ(stepInner.npuMemUsages_[0].stepMaxAllocated, 100);
    ASSERT_EQ(stepInner.npuMemUsages_[0].stepMinAllocated, 20);
}

TEST(StepInnerAnalyzerUpdateAllocatedFuncTest, UpdateAllocatedInitTest)
{
    Config config;
    BitField<decltype(config.eventType)> eventBit;
    BitField<decltype(config.analysisType)> analysisBit;
    analysisBit.setBit(static_cast<size_t>(AnalysisType::LEAKS_ANALYSIS));
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.analysisType = analysisBit.getValue();
    config.eventType = eventBit.getValue();
    config.stepList.stepCount = 0;
    StepInnerAnalyzer stepInner{config};
    NpuMemUsage npumemusage;
    npumemusage.mstxStep = 2;

    npumemusage.stepMaxAllocated = 0;
    npumemusage.stepMinAllocated = 0;
    stepInner.npuMemUsages_.insert({0, npumemusage});
    stepInner.UpdateAllocated(0, 100);
    stepInner.UpdateAllocated(0, 200);
    ASSERT_EQ(stepInner.npuMemUsages_[0].stepMaxAllocated, 200);
    ASSERT_EQ(stepInner.npuMemUsages_[0].stepMinAllocated, 100);
}

TEST(StepInnerAnalyzerUpdateAllocatedFuncTest, UpdateAllocatedreturnTest)
{
    Config config;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.eventType = eventBit.getValue();
    config.stepList.stepCount = 0;
    StepInnerAnalyzer stepInner{config};
    NpuMemUsage npumemusage;
    npumemusage.mstxStep = 0;

    npumemusage.stepMaxAllocated = 0;
    npumemusage.stepMinAllocated = 0;
    stepInner.npuMemUsages_.insert({0, npumemusage});
    stepInner.UpdateAllocated(0, 100);
    stepInner.UpdateAllocated(0, 200);
    ASSERT_EQ(stepInner.npuMemUsages_[0].stepMaxAllocated, 0);
    ASSERT_EQ(stepInner.npuMemUsages_[0].stepMinAllocated, 0);
}

TEST(StepInnerAnalyzerAddDurationTest, AddDurationTest)
{
    Config config;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.eventType = eventBit.getValue();
    config.stepList.stepCount = 0;
    StepInnerAnalyzer stepInner{config};
    NpuMemUsage npumemusage;
    npumemusage.mstxStep = 0;

    Leaks::NpuMemInfo memInfo;
    memInfo.duration = 1;
    npumemusage.mempooltable.insert({0, memInfo});
    stepInner.npuMemUsages_.insert({0, npumemusage});
    stepInner.AddDuration(0);
    ASSERT_EQ(stepInner.npuMemUsages_[0].mempooltable[0].duration, 2);
}

TEST(StepInnerAnalyzerAddDurationTest, AddDurationReturnTest)
{
    Config config;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.eventType = eventBit.getValue();
    config.stepList.stepCount = 0;
    StepInnerAnalyzer stepInner{config};
    NpuMemUsage npumemusage;
    npumemusage.mstxStep = 0;

    Leaks::NpuMemInfo memInfo;
    memInfo.duration = 1;
    npumemusage.mempooltable.insert({0, memInfo});
    stepInner.npuMemUsages_.insert({1, npumemusage});
    stepInner.AddDuration(0); // 不存在的deviceID，提前返回
    ASSERT_EQ(stepInner.npuMemUsages_[1].mempooltable[0].duration, 1);
}

TEST(StepInnerAnalyzerSetStepIdFuncTest, SetStepIdTest)
{
    Config config;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.eventType = eventBit.getValue();
    config.stepList.stepCount = 0;
    StepInnerAnalyzer stepInner{config};
    NpuMemUsage npumemusage;
    npumemusage.mstxStep = 1;

    stepInner.npuMemUsages_.insert({1, npumemusage});
    stepInner.SetStepId(1, 2);
    ASSERT_EQ(stepInner.npuMemUsages_[1].mstxStep, 2);
}

TEST(StepInnerAnalyzerGetNowAllocatedFuncTest, GetNowAllocatedTest)
{
    Config config;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    config.eventType = eventBit.getValue();
    config.stepList.stepCount = 0;
    StepInnerAnalyzer stepInner{config};
    NpuMemUsage npumemusage;
    npumemusage.mstxStep = 1;
    npumemusage.totalAllocated = 500;

    stepInner.npuMemUsages_.insert({1, npumemusage});
    
    ASSERT_EQ(stepInner.GetNowAllocated(1), 500);
}