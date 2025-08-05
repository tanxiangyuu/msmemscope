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
    auto buffer = RecordBuffer::CreateRecordBuffer<MstxRecord>(TLVBlockType::MARK_MESSAGE, message);
    MstxRecord* mstxRecord = buffer.Cast<MstxRecord>();
    mstxRecord->markType = type;
    mstxRecord->devId = 0;
    mstxRecord->stepId = stepId;
    mstxRecord->rangeId = rangeId;
    mstxRecord->streamId = streamId;
    return *mstxRecord;
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

    MemPoolRecord record1;
    record1.type = RecordType::PTA_CACHING_POOL_RECORD;
    record1.recordIndex = 1;
    auto memoryusage1 = MemoryUsage {};
    memoryusage1.deviceIndex = 0;
    memoryusage1.dataType = 0;
    memoryusage1.ptr = 12345;
    memoryusage1.allocSize = 512;
    memoryusage1.totalAllocated = 512;
    record1.memoryUsage = memoryusage1;

    MemPoolRecord record2;
    record2.type = RecordType::PTA_CACHING_POOL_RECORD;
    record2.recordIndex = 2;
    auto memoryusage2 = MemoryUsage {};
    memoryusage2.deviceIndex = 0;
    memoryusage2.dataType = 1;
    memoryusage2.ptr = 12345;
    memoryusage2.allocSize = 512;
    memoryusage2.totalAllocated = 0;
    record2.memoryUsage = memoryusage2;

    EXPECT_TRUE(StepInnerAnalyzer::GetInstance(analysisConfig).Record(clientId, record1));
    EXPECT_TRUE(StepInnerAnalyzer::GetInstance(analysisConfig).Record(clientId, record2));
}

TEST(StepInnerAnalyzerTest, do_reveive_mstxmsg_expect_leaks_warning)
{
    ClientId clientId = 0;
    auto mstxRecordStartBuf1 = RecordBuffer::CreateRecordBuffer<MstxRecord>(TLVBlockType::MARK_MESSAGE, "step start");
    MstxRecord* mstxRecordStart1 = mstxRecordStartBuf1.Cast<MstxRecord>();
    mstxRecordStart1->markType = MarkType::RANGE_START_A;
    mstxRecordStart1->devId = 0;
    mstxRecordStart1->stepId = 1;
    mstxRecordStart1->streamId = 123;

    // 经过第二个step，但仍然未释放
    auto mstxRecordStartBuf2 = RecordBuffer::CreateRecordBuffer<MstxRecord>(TLVBlockType::MARK_MESSAGE, "step start");
    MstxRecord* mstxRecordStart2 = mstxRecordStartBuf2.Cast<MstxRecord>();
    mstxRecordStart2->markType = MarkType::RANGE_START_A;
    mstxRecordStart2->devId = 0;
    mstxRecordStart2->stepId = 2;
    mstxRecordStart2->streamId = 123;

    auto mstxRecordEndBuf = RecordBuffer::CreateRecordBuffer<MstxRecord>();
    MstxRecord* mstxRecordEnd = mstxRecordEndBuf.Cast<MstxRecord>();
    mstxRecordEnd->markType = MarkType::RANGE_END;
    mstxRecordEnd->devId = 0;
    mstxRecordEnd->stepId = 2;
    mstxRecordEnd->streamId = 123;

    // 经过两个step的内存
    auto buffer1 = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* record1 = buffer1.Cast<MemPoolRecord>();
    record1->type = RecordType::PTA_CACHING_POOL_RECORD;
    record1->recordIndex = 1;
    auto memoryusage1 = MemoryUsage {};
    memoryusage1.deviceIndex = 0;
    memoryusage1.dataType = 0;
    memoryusage1.ptr = 12345;
    memoryusage1.allocSize = 512;
    memoryusage1.totalAllocated = 512;
    record1->memoryUsage = memoryusage1;

    // 先初始化注册
    Config analysisConfig;
    BitField<decltype(analysisConfig.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    analysisConfig.eventType = eventBit.getValue();
    analysisConfig.stepList.stepCount = 0;
    StepInnerAnalyzer::GetInstance(analysisConfig).config_.stepList.stepCount = 0;
    static StepInnerAnalyzer analyzer(analysisConfig);
    EXPECT_TRUE(StepInnerAnalyzer::GetInstance(analysisConfig).Record(clientId, *record1));
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(*mstxRecordStart1);
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(*mstxRecordStart2);
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(*mstxRecordEnd);
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

    auto buffer = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* record = buffer.Cast<MemPoolRecord>();
    record->type = RecordType::PTA_CACHING_POOL_RECORD;
    record->recordIndex = 1;
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
    record->memoryUsage = memoryusage;

    EXPECT_TRUE(StepInnerAnalyzer::GetInstance(analysisConfig).Record(clientId, *record));
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

    auto buffer = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* record = buffer.Cast<MemPoolRecord>();
    record->type = RecordType::PTA_CACHING_POOL_RECORD;
    record->recordIndex = 1;
    auto memoryusage = MemoryUsage {};
    memoryusage.deviceIndex = 0;
    memoryusage.dataType = 0;
    memoryusage.ptr = 12345;
    memoryusage.allocSize = 512;
    record->memoryUsage = memoryusage;

    // 地址重复的申请
    auto double_record_buffer = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* double_record = double_record_buffer.Cast<MemPoolRecord>();
    double_record->type = RecordType::PTA_CACHING_POOL_RECORD;
    double_record->recordIndex = 2;
    auto double_memoryusage = MemoryUsage {};
    double_memoryusage.deviceIndex = 0;
    double_memoryusage.dataType = 0;
    double_memoryusage.ptr = 12345;
    double_memoryusage.allocSize = 512;
    double_record->memoryUsage = double_memoryusage;

    EXPECT_TRUE(StepInnerAnalyzer::GetInstance(analysisConfig).Record(clientId, *record));
    EXPECT_TRUE(StepInnerAnalyzer::GetInstance(analysisConfig).Record(clientId, *double_record));
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

    auto buffer = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* record = buffer.Cast<MemPoolRecord>();
    record->type = RecordType::PTA_CACHING_POOL_RECORD;
    record->recordIndex = 2;
    auto memoryusage = MemoryUsage {};
    memoryusage.deviceType = 20;
    memoryusage.deviceIndex = 0;
    memoryusage.dataType = 1;
    memoryusage.allocatorType = 0;
    memoryusage.ptr = 12345;
    memoryusage.allocSize = 512;
    memoryusage.totalAllocated = 0;
    memoryusage.totalActive = 0;
    memoryusage.totalReserved = 1024;
    memoryusage.streamPtr = 4321;
    record->memoryUsage = memoryusage;

    EXPECT_TRUE(StepInnerAnalyzer::GetInstance(analysisConfig).Record(clientId, *record));
}

TEST(StepInnerAnalyzerTest, do_reveive_mstxmsg_expect_torch_leaks) {
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

    // 构造第四个step时仍未释放的情景，超过duration限制
    auto mstxRecordStartFourth = CreatMstxRecord(MarkType::RANGE_START_A, "step start", 4, 4, 123);
    auto mstxRecordEndFourth = CreatMstxRecord(MarkType::RANGE_END, "", 4, 4, 123);

    // step前后allocated内存不一致
    auto buffer = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* record = buffer.Cast<MemPoolRecord>();
    record->type = RecordType::PTA_CACHING_POOL_RECORD;
    record->recordIndex = 1;
    record->type = RecordType::PTA_CACHING_POOL_RECORD;
    auto memoryusage = MemoryUsage {};
    memoryusage.deviceIndex = 0;
    memoryusage.dataType = 0;
    memoryusage.ptr = 92345;
    memoryusage.allocSize = 512;
    memoryusage.totalAllocated = 512;
    record->memoryUsage = memoryusage;

    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordStartFirst);
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordEndFirst);
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordStartSecond);
    EXPECT_TRUE(StepInnerAnalyzer::GetInstance(analysisConfig).Record(clientId, *record));
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordEndSecond);
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordStartThird);
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordEndThird);
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordStartFourth);
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordEndFourth);
}

TEST(StepInnerAnalyzerTest, do_reveive_mstxmsg_expect_mindspore_leaks) {
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

    auto mstxRecordStartFirst = CreatMstxRecord(MarkType::RANGE_START_A, "step start", 1, 1, 123);
    auto mstxRecordEndFirst = CreatMstxRecord(MarkType::RANGE_END, "", 1, 1, 123);
    auto mstxRecordStartSecond = CreatMstxRecord(MarkType::RANGE_START_A, "step start", 2, 2, 123);
    auto mstxRecordEndSecond = CreatMstxRecord(MarkType::RANGE_END, "", 2, 2, 123);
    auto mstxRecordStartThird = CreatMstxRecord(MarkType::RANGE_START_A, "step start", 3, 3, 123);
    auto mstxRecordEndThird = CreatMstxRecord(MarkType::RANGE_END, "", 3, 3, 123);
    auto mstxRecordStartFourth = CreatMstxRecord(MarkType::RANGE_START_A, "step start", 4, 4, 123);
    auto mstxRecordEndFourth = CreatMstxRecord(MarkType::RANGE_END, "", 4, 4, 123);

    // step前后allocated内存不一致
    auto buffer = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* record = buffer.Cast<MemPoolRecord>();
    record->type = RecordType::MINDSPORE_NPU_RECORD;
    record->recordIndex = 1;
    auto memoryusage = MemoryUsage {};
    memoryusage.deviceIndex = 0;
    memoryusage.dataType = 0;
    memoryusage.ptr = 93345;
    memoryusage.allocSize = 512;
    memoryusage.totalAllocated = 512;
    record->memoryUsage = memoryusage;

    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordStartFirst);
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordEndFirst);
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordStartSecond);
    EXPECT_TRUE(StepInnerAnalyzer::GetInstance(analysisConfig).Record(clientId, *record));
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordEndSecond);
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordStartThird);
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordEndThird);
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordStartFourth);
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordEndFourth);
}

TEST(StepInnerAnalyzerTest, do_reveive_mstxmsg_expect_atb_leaks) {
    ClientId clientId = 0;
    Leaks::DeviceId deviceId = 0;
    Config analysisConfig;
    BitField<decltype(analysisConfig.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    analysisConfig.eventType = eventBit.getValue();
    analysisConfig.stepList.stepCount = 0;
    StepInnerAnalyzer::GetInstance(analysisConfig).config_.stepList.stepCount = 0;
    static StepInnerAnalyzer analyzer(analysisConfig);

    auto mstxRecordStartFirst = CreatMstxRecord(MarkType::RANGE_START_A, "step start", 1, 1, 123);
    auto mstxRecordEndFirst = CreatMstxRecord(MarkType::RANGE_END, "", 1, 1, 123);
    auto mstxRecordStartSecond = CreatMstxRecord(MarkType::RANGE_START_A, "step start", 2, 2, 123);
    auto mstxRecordEndSecond = CreatMstxRecord(MarkType::RANGE_END, "", 2, 2, 123);
    auto mstxRecordStartThird = CreatMstxRecord(MarkType::RANGE_START_A, "step start", 3, 3, 123);
    auto mstxRecordEndThird = CreatMstxRecord(MarkType::RANGE_END, "", 3, 3, 123);
    auto mstxRecordStartFourth = CreatMstxRecord(MarkType::RANGE_START_A, "step start", 4, 4, 123);
    auto mstxRecordEndFourth = CreatMstxRecord(MarkType::RANGE_END, "", 4, 4, 123);

    // step前后allocated内存不一致
    auto buffer = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* record = buffer.Cast<MemPoolRecord>();
    record->type = RecordType::ATB_MEMORY_POOL_RECORD;
    auto npuRecordMalloc = MemPoolRecord {};
    record->recordIndex = 1;
    record->type = RecordType::ATB_MEMORY_POOL_RECORD;
    auto memoryusage = MemoryUsage {};
    memoryusage.deviceIndex = 0;
    memoryusage.dataType = 0;
    memoryusage.ptr = 94345;
    memoryusage.allocSize = 512;
    memoryusage.totalAllocated = 512;
    record->memoryUsage = memoryusage;

    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordStartFirst);
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordEndFirst);
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordStartSecond);
    EXPECT_TRUE(StepInnerAnalyzer::GetInstance(analysisConfig).Record(clientId, *record));
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordEndSecond);
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordStartThird);
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordEndThird);
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordStartFourth);
    StepInnerAnalyzer::GetInstance(analysisConfig).ReceiveMstxMsg(mstxRecordEndFourth);
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

TEST(StepInnerAnalyzerTest, set_collect_mode_to_custom_disable_analysis)
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
    config.collectMode = 1;
    StepInnerAnalyzer stepInner{config};
    auto ret = stepInner.IsStepInnerAnalysisEnable();
    ASSERT_FALSE(ret);
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
    npumemusage.poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMaxAllocated = 0;
    npumemusage.poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMinAllocated = 0;
    stepInner.npuMemUsages_.insert({0, npumemusage});
    stepInner.UpdateAllocated(0, RecordType::PTA_CACHING_POOL_RECORD, 100);
    ASSERT_EQ(stepInner.npuMemUsages_[0].poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMaxAllocated, 0);
    ASSERT_EQ(stepInner.npuMemUsages_[0].poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMinAllocated, 0);
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

    npumemusage.poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMaxAllocated = 20;
    npumemusage.poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMinAllocated = 20;
    stepInner.npuMemUsages_.insert({0, npumemusage});
    stepInner.UpdateAllocated(0, RecordType::PTA_CACHING_POOL_RECORD, 100);
    ASSERT_EQ(stepInner.npuMemUsages_[0].poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMaxAllocated, 20);
    ASSERT_EQ(stepInner.npuMemUsages_[0].poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMinAllocated, 20);
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
    npumemusage.poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMaxAllocated = 100;
    npumemusage.poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMinAllocated = 20;
    stepInner.npuMemUsages_.insert({0, npumemusage});
    stepInner.CheckGap(0);
    ASSERT_EQ(stepInner.npuMemUsages_[0].poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMaxAllocated, 0);
    ASSERT_EQ(stepInner.npuMemUsages_[0].poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMinAllocated, 0);
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
    npumemusage.poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMaxAllocated = 100;
    npumemusage.poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMinAllocated = 20;
    npumemusage.poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].maxGapInfo = gapinfo;
    stepInner.npuMemUsages_.insert({0, npumemusage});
    stepInner.CheckGap(0);
    ASSERT_EQ(stepInner.npuMemUsages_[0].poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMaxAllocated, 0);
    ASSERT_EQ(stepInner.npuMemUsages_[0].poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMinAllocated, 0);
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
    Leaks::RecordBase record;
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

    MemPoolRecord record1;
    record1.type = RecordType::PTA_CACHING_POOL_RECORD;
    record1.recordIndex = 1;
    auto memoryusage1 = MemoryUsage {};
    memoryusage1.deviceIndex = 0;
    memoryusage1.dataType = 0;
    memoryusage1.ptr = 12345;
    memoryusage1.allocSize = 512;
    memoryusage1.totalAllocated = 512;
    record1.memoryUsage = memoryusage1;

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

    auto buffer = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* record1 = buffer.Cast<MemPoolRecord>();
    record1->type = RecordType::PTA_CACHING_POOL_RECORD;
    record1->recordIndex = 1;
    auto memoryusage1 = MemoryUsage {};
    memoryusage1.deviceIndex = 0;
    memoryusage1.dataType = 1;
    memoryusage1.ptr = 12345;
    memoryusage1.allocSize = 512;
    memoryusage1.totalAllocated = 512;
    record1->memoryUsage = memoryusage1;

    EXPECT_TRUE(StepInnerAnalyzer::GetInstance(config).Record(clientId, *record1));
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

    auto buffer = RecordBuffer::CreateRecordBuffer<MstxRecord>(TLVBlockType::MARK_MESSAGE, "step start");
    MstxRecord* mstxRecordStart1 = buffer.Cast<MstxRecord>();
    mstxRecordStart1->markType = MarkType::RANGE_START_A;
    mstxRecordStart1->devId = 0;
    mstxRecordStart1->stepId = 1;
    mstxRecordStart1->streamId = 123;

    StepInnerAnalyzer::GetInstance(config).ReceiveMstxMsg(*mstxRecordStart1);
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

    auto buffer = RecordBuffer::CreateRecordBuffer<MstxRecord>(TLVBlockType::MARK_MESSAGE, "step end");
    MstxRecord* mstxRecordStart1 = buffer.Cast<MstxRecord>();
    mstxRecordStart1->markType = Leaks::MarkType::RANGE_END;
    mstxRecordStart1->devId = 0;
    mstxRecordStart1->stepId = 1;
    mstxRecordStart1->streamId = 123;

    StepInnerAnalyzer::GetInstance(config).ReceiveMstxMsg(*mstxRecordStart1);
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

    npumemusage.poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMaxAllocated = 20;
    npumemusage.poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMinAllocated = 20;
    stepInner.npuMemUsages_.insert({0, npumemusage});
    stepInner.UpdateAllocated(0, RecordType::PTA_CACHING_POOL_RECORD, 100);
    ASSERT_EQ(stepInner.npuMemUsages_[0].poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMaxAllocated, 100);
    ASSERT_EQ(stepInner.npuMemUsages_[0].poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMinAllocated, 20);
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

    npumemusage.poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMaxAllocated = 0;
    npumemusage.poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMinAllocated = 0;
    stepInner.npuMemUsages_.insert({0, npumemusage});
    stepInner.UpdateAllocated(0, RecordType::PTA_CACHING_POOL_RECORD, 100);
    stepInner.UpdateAllocated(0, RecordType::PTA_CACHING_POOL_RECORD, 200);
    ASSERT_EQ(stepInner.npuMemUsages_[0].poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMaxAllocated, 200);
    ASSERT_EQ(stepInner.npuMemUsages_[0].poolStatusTable[RecordType::PTA_CACHING_POOL_RECORD].stepMinAllocated, 100);
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

    npumemusage.poolStatusTable[RecordType::MINDSPORE_NPU_RECORD].stepMaxAllocated = 0;
    npumemusage.poolStatusTable[RecordType::MINDSPORE_NPU_RECORD].stepMinAllocated = 0;
    stepInner.npuMemUsages_.insert({0, npumemusage});
    stepInner.UpdateAllocated(0, RecordType::MINDSPORE_NPU_RECORD, 100);
    stepInner.UpdateAllocated(0, RecordType::MINDSPORE_NPU_RECORD, 200);
    ASSERT_EQ(stepInner.npuMemUsages_[0].poolStatusTable[RecordType::MINDSPORE_NPU_RECORD].stepMaxAllocated, 0);
    ASSERT_EQ(stepInner.npuMemUsages_[0].poolStatusTable[RecordType::MINDSPORE_NPU_RECORD].stepMinAllocated, 0);
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
    npumemusage.poolOpTable.insert({Leaks::NpuMemKey(0, RecordType::PTA_CACHING_POOL_RECORD), memInfo});
    stepInner.npuMemUsages_.insert({0, npumemusage});
    stepInner.AddDuration(0);
    ASSERT_EQ(stepInner.npuMemUsages_[0].poolOpTable[Leaks::NpuMemKey(0, RecordType::PTA_CACHING_POOL_RECORD)].duration, 2);
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
    npumemusage.poolOpTable.insert({Leaks::NpuMemKey(0, RecordType::PTA_CACHING_POOL_RECORD), memInfo});
    stepInner.npuMemUsages_.insert({1, npumemusage});
    stepInner.AddDuration(0); // 不存在的deviceID，提前返回
    ASSERT_EQ(stepInner.npuMemUsages_[1].poolOpTable[Leaks::NpuMemKey(0, RecordType::PTA_CACHING_POOL_RECORD)].duration, 1);
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