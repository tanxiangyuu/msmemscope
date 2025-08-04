// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#define private public
#include "event_trace/trace_manager/event_trace_manager.h"
#include "event_trace/event_report.h"
#undef private
#include "event_trace/vallina_symbol.h"
#include "securec.h"
#include "bit_field.h"

using namespace Leaks;

TEST(EventReportTest, EventReportInstanceTest) {
    EventReport& instance1 = EventReport::Instance(CommType::MEMORY);
    BitField<decltype(instance1.config_.eventType)> eventBit;
    BitField<decltype(instance1.config_.levelType)> levelBit;
    levelBit.setBit(static_cast<size_t>(LevelType::LEVEL_OP));
    levelBit.setBit(static_cast<size_t>(LevelType::LEVEL_KERNEL));
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::LAUNCH_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::ACCESS_EVENT));
    instance1.config_.eventType = eventBit.getValue();
    instance1.config_.levelType = levelBit.getValue();
    instance1.config_.enableCStack = true;
    instance1.config_.enablePyStack = true;
    EventReport& instance2 = EventReport::Instance(CommType::MEMORY);
    EXPECT_EQ(&instance1, &instance2);
}

TEST(EventReportTest, ReportMallocTestDEVICE) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    BitField<decltype(instance.config_.eventType)> eventBit;
    BitField<decltype(instance.config_.levelType)> levelBit;
    levelBit.setBit(static_cast<size_t>(LevelType::LEVEL_OP));
    levelBit.setBit(static_cast<size_t>(LevelType::LEVEL_KERNEL));
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::LAUNCH_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::ACCESS_EVENT));
    instance.config_.eventType = eventBit.getValue();
    instance.config_.levelType = levelBit.getValue();
    instance.config_.enableCStack = true;
    instance.config_.enablePyStack = true;
    uint64_t testAddr = 0x12345678;
    uint64_t testSize = 1024;
    unsigned long long testFlag = 2377900603261207558;
    MemOpSpace space = MemOpSpace::DEVICE;
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectAllNpu = true;
    CallStackString callStack;
    EXPECT_TRUE(instance.ReportMalloc(testAddr, testSize, 1, callStack));
}

TEST(EventReportTest, ReportAddrInfoTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    auto buffer = RecordBuffer::CreateRecordBuffer<AddrInfo>();
    EXPECT_TRUE(instance.ReportAddrInfo(buffer));
}

TEST(EventReportTest, ReportTorchNpuMallocTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    auto buffer = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* npuRecordMalloc = buffer.Cast<MemPoolRecord>();
    npuRecordMalloc->recordIndex = 1;
    auto memoryusage1 = MemoryUsage {};
    memoryusage1.deviceIndex = 0;
    memoryusage1.dataType = 0;
    memoryusage1.ptr = 12345;
    memoryusage1.allocSize = 512;
    memoryusage1.totalAllocated = 512;
    npuRecordMalloc->memoryUsage = memoryusage1;
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectAllNpu = true;
    EXPECT_TRUE(instance.ReportMemPoolRecord(buffer));
}

TEST(EventReportTest, ReportTorchNpuFreeTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    auto buffer = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* npuRecordFree = buffer.Cast<MemPoolRecord>();
    npuRecordFree->recordIndex = 3;
    auto memoryusage1 = MemoryUsage {};
    memoryusage1.deviceIndex = 3;
    memoryusage1.dataType = 1;
    memoryusage1.ptr = 12345;
    memoryusage1.allocSize = 512;
    memoryusage1.totalAllocated = 512;
    npuRecordFree->memoryUsage = memoryusage1;
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectAllNpu = true;
    EXPECT_TRUE(instance.ReportMemPoolRecord(buffer));
}

TEST(EventReportTest, ReportTorchNpuConditionTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    auto buffer = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* npuRecordFree = buffer.Cast<MemPoolRecord>();
    npuRecordFree->recordIndex = 3;
    auto memoryusage1 = MemoryUsage {};
    memoryusage1.deviceIndex = 3;
    memoryusage1.dataType = 1;
    memoryusage1.ptr = 12345;
    memoryusage1.allocSize = 512;
    memoryusage1.totalAllocated = 512;
    npuRecordFree->memoryUsage = memoryusage1;
    instance.isReceiveServerInfo_ = true;

    EventTraceManager::Instance().SetTraceStatus(EventTraceStatus::NOT_IN_TRACING);
    EXPECT_TRUE(instance.ReportMemPoolRecord(buffer));
    EventTraceManager::Instance().SetTraceStatus(EventTraceStatus::IN_TRACING);
    instance.config_.collectAllNpu = false;
    EXPECT_TRUE(instance.ReportMemPoolRecord(buffer));
    instance.config_.collectAllNpu = true;
}

TEST(EventReportTest, ReportMallocTestHost) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    BitField<decltype(instance.config_.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::LAUNCH_EVENT));
    instance.config_.eventType = eventBit.getValue();
    instance.config_.enableCStack = true;
    instance.config_.enablePyStack = true;
    uint64_t testAddr = 0x12345678;
    uint64_t testSize = 1024;
    unsigned long long testFlag = 504403158274934784;
    MemOpSpace space = MemOpSpace::HOST;
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectAllNpu = true;
    CallStackString callStack;
    EXPECT_TRUE(instance.ReportMalloc(testAddr, testSize, 1, callStack));
}

TEST(EventReportTest, ReportFreeTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    BitField<decltype(instance.config_.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::LAUNCH_EVENT));
    instance.config_.eventType = eventBit.getValue();
    instance.config_.enableCStack = true;
    instance.config_.enablePyStack = true;
    uint64_t testAddr = 0x12345678;
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectAllNpu = true;
    CallStackString callStack;
    EXPECT_TRUE(instance.ReportFree(testAddr, callStack));
}

TEST(EventReportTest, ReportHostMallocWithoutMstxTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    BitField<decltype(instance.config_.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::LAUNCH_EVENT));
    instance.config_.eventType = eventBit.getValue();
    instance.config_.enableCStack = true;
    instance.config_.enablePyStack = true;
    uint64_t testAddr = 0x12345678;
    uint64_t testSize = 1024;
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectCpu = true;
    CallStackString callStack;
    EXPECT_TRUE(instance.ReportHostMalloc(testAddr, testSize));
}
 
TEST(EventReportTest, ReportHostFreeWithoutMstxTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    BitField<decltype(instance.config_.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::LAUNCH_EVENT));
    instance.config_.eventType = eventBit.getValue();
    uint64_t testAddr = 0x12345678;
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectAllNpu = true;
    CallStackString callStack;
    EXPECT_TRUE(instance.ReportHostFree(testAddr));
}

TEST(EventReportTest, ReportHostMallocTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    BitField<decltype(instance.config_.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::LAUNCH_EVENT));
    instance.config_.eventType = eventBit.getValue();
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectCpu = true;
    auto buffer1 = RecordBuffer::CreateRecordBuffer<MstxRecord>(
        TLVBlockType::MARK_MESSAGE, "report host memory info start");
    MstxRecord* mstxRecordStart = buffer1.Cast<MstxRecord>();
    mstxRecordStart->markType = MarkType::RANGE_START_A;
    mstxRecordStart->rangeId = 1;
    instance.ReportMark(buffer1);

    uint64_t testAddr = 0x12345678;
    uint64_t testSize = 1024;
    EXPECT_TRUE(instance.ReportHostMalloc(testAddr, testSize));

    auto buffer2 = RecordBuffer::CreateRecordBuffer<MstxRecord>();
    MstxRecord* mstxRecordEnd = buffer2.Cast<MstxRecord>();
    mstxRecordEnd->markType = MarkType::RANGE_END;
    mstxRecordEnd->rangeId = 1;
    instance.ReportMark(buffer2);
}
 
TEST(EventReportTest, ReportHostFreeTest) {
    EventReport &instance = EventReport::Instance(CommType::MEMORY);
    BitField<decltype(instance.config_.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::LAUNCH_EVENT));
    instance.config_.eventType = eventBit.getValue();
    CallStackString callStack;
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectAllNpu = true;
    auto buffer1 = RecordBuffer::CreateRecordBuffer<MstxRecord>(
        TLVBlockType::MARK_MESSAGE, "report host memory info start");
    MstxRecord* mstxRecordStart = buffer1.Cast<MstxRecord>();
    mstxRecordStart->markType = MarkType::RANGE_START_A;
    mstxRecordStart->rangeId = 1;
    instance.ReportMark(buffer1);
    uint64_t testAddr = 0x12345678;
    EXPECT_TRUE(instance.ReportHostFree(testAddr));

    auto buffer2 = RecordBuffer::CreateRecordBuffer<MstxRecord>();
    MstxRecord* mstxRecordEnd = buffer2.Cast<MstxRecord>();
    mstxRecordEnd->markType = MarkType::RANGE_END;
    mstxRecordEnd->rangeId = 1;
    instance.ReportMark(buffer2);
}

TEST(EventReportTest, ReportMarkTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectAllNpu = true;
    auto buffer = RecordBuffer::CreateRecordBuffer<MstxRecord>();
    MstxRecord* record = buffer.Cast<MstxRecord>();
    record->rangeId = 123;
    EXPECT_TRUE(instance.ReportMark(buffer));
}

TEST(EventReportTest, ReportKernelLaunchTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    BitField<decltype(instance.config_.eventType)> eventBit;
    BitField<decltype(instance.config_.levelType)> levelBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::LAUNCH_EVENT));
    levelBit.setBit(static_cast<size_t>(LevelType::LEVEL_KERNEL));
    instance.config_.eventType = eventBit.getValue();
    instance.config_.levelType = levelBit.getValue();
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectAllNpu = true;
    int16_t devId = 1;
    int16_t streamId = 1;
    int16_t taskId = 1;
    auto taskKey = std::make_tuple(devId, streamId, taskId);
    AclnnKernelMapInfo kernelLaunchInfo {};
    kernelLaunchInfo.taskKey = taskKey;
    kernelLaunchInfo.timestamp = 123;
    kernelLaunchInfo.kernelName = "add";
    EXPECT_TRUE(instance.ReportKernelLaunch(kernelLaunchInfo));
}

TEST(EventReportTest, ReportAclItfTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectAllNpu = true;
    EXPECT_TRUE(instance.ReportAclItf(RecordSubType::INIT));
}

TEST(EventReportTest, ReportAtbOpExecuteTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectAllNpu = true;
    RecordBuffer atbOpExecuteRecord = RecordBuffer::CreateRecordBuffer<AtbOpExecuteRecord>();
    EXPECT_TRUE(instance.ReportAtbOpExecute(atbOpExecuteRecord));
}

TEST(EventReportTest, ReportAtbKernelTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectAllNpu = true;
    RecordBuffer atbKernelRecord = RecordBuffer::CreateRecordBuffer<AtbKernelRecord>();
    EXPECT_TRUE(instance.ReportAtbKernel(atbKernelRecord));
}

TEST(EventReportTest, ReportAtbMemAccessTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectAllNpu = true;
    RecordBuffer memAccessRecord = RecordBuffer::CreateRecordBuffer<MemAccessRecord>();
    std::vector<RecordBuffer> records;
    records.push_back(memAccessRecord);
    EXPECT_TRUE(instance.ReportAtbAccessMemory(records));
}

TEST(EventReportTest, ReportAtenLaunchTestExpectSuccess)
{
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectAllNpu = true;
    auto atenOpLaunchRecord = RecordBuffer::CreateRecordBuffer<AtenOpLaunchRecord>();
    EXPECT_TRUE(instance.ReportAtenLaunch(atenOpLaunchRecord));
}

TEST(EventReportTest, ReportAtenAccessTestExpectSuccess)
{
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectAllNpu = true;
    auto memAccessRecord = RecordBuffer::CreateRecordBuffer<MemAccessRecord>();
    EXPECT_TRUE(instance.ReportAtenAccess(memAccessRecord));
}

TEST(EventReportTest, ReportAtenLaunchTestExpextSuccess)
{
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectAllNpu = true;
    auto atenOpLaunchRecord = RecordBuffer::CreateRecordBuffer<AtenOpLaunchRecord>();
    EXPECT_TRUE(instance.ReportAtenLaunch(atenOpLaunchRecord));
}

TEST(EventReportTest, ReportAtenAccessTestExpextSuccess)
{
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectAllNpu = true;
    auto memAccessRecord = RecordBuffer::CreateRecordBuffer<MemAccessRecord>();
    EXPECT_TRUE(instance.ReportAtenAccess(memAccessRecord));
}

TEST(EventReportTest, ReportKernelExcuteTestExpextSuccess)
{
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectAllNpu = true;
    MemAccessRecord  memAccessRecord  {};
    CallStackString stack;
    std::string name = "test";
    uint64_t time = 1234;
    RecordSubType type = RecordSubType::KERNEL_START;
    TaskKey key;
    EXPECT_TRUE(instance.ReportKernelExcute(key, name, time, type));
}

struct TestLoader {
    static void *Load(void)
    {
        void *ptr = nullptr;
        return ptr;
    }
};

TEST(EventReportTest, VallinaSymbolTest) {

    using va = VallinaSymbol<TestLoader>;
    char const *symbol;
    va::Instance().Get(symbol);
}

void ResetEventReportStepInfo()
{
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.stepInfo_.currentStepId = 0;
    instance.stepInfo_.inStepRange = false;
    instance.stepInfo_.stepMarkRangeIdList.clear();
    return;
}

TEST(EventReportTest, TestReportSkipStepsNormal)
{
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectAllNpu = true;
    auto buffer = RecordBuffer::CreateRecordBuffer<MstxRecord>(TLVBlockType::MARK_MESSAGE, "step start");
    MstxRecord* mstxRecord = buffer.Cast<MstxRecord>();
    mstxRecord->markType = MarkType::RANGE_START_A;
    mstxRecord->rangeId = 8;
    instance.SetStepInfo(*mstxRecord);
    instance.config_.stepList.stepCount = 3;
    instance.config_.stepList.stepIdList[0] = 1;
    instance.config_.stepList.stepIdList[1] = 2;
    instance.config_.stepList.stepIdList[2] = 6;
    instance.config_.collectAllNpu = true;
    EXPECT_EQ(instance.IsNeedSkip(GD_INVALID_NUM), false);

    mstxRecord->markType = MarkType::RANGE_END;
    instance.SetStepInfo(*mstxRecord);
    EXPECT_EQ(instance.IsNeedSkip(GD_INVALID_NUM), true);

    mstxRecord->markType = MarkType::RANGE_START_A;
    mstxRecord->rangeId = 9;
    instance.SetStepInfo(*mstxRecord);
    EXPECT_EQ(instance.IsNeedSkip(GD_INVALID_NUM), false);

    mstxRecord->markType = MarkType::RANGE_END;
    instance.SetStepInfo(*mstxRecord);
    EXPECT_EQ(instance.IsNeedSkip(GD_INVALID_NUM), true);

    mstxRecord->markType = MarkType::RANGE_START_A;
    mstxRecord->rangeId = 10;
    instance.SetStepInfo(*mstxRecord);
    EXPECT_EQ(instance.IsNeedSkip(GD_INVALID_NUM), true);

    mstxRecord->markType = MarkType::RANGE_END;
    instance.SetStepInfo(*mstxRecord);
    EXPECT_EQ(instance.IsNeedSkip(GD_INVALID_NUM), true);

    ResetEventReportStepInfo();
}

TEST(EventReportTest, TestReportSkipStepsWithNoMstx)
{
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectAllNpu = true;
    instance.config_.stepList.stepCount = 3;
    instance.config_.stepList.stepIdList[0] = 1;
    instance.config_.stepList.stepIdList[1] = 2;
    instance.config_.stepList.stepIdList[2] = 6;
    EXPECT_EQ(instance.IsNeedSkip(GD_INVALID_NUM), true);

    ResetEventReportStepInfo();
}

TEST(EventReportTest, TestReportSkipStepsWithOtherMessageMstx)
{
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectAllNpu = true;
    auto buffer = RecordBuffer::CreateRecordBuffer<MstxRecord>(
        TLVBlockType::MARK_MESSAGE, "report host memory info start");
    MstxRecord* mstxRecord = buffer.Cast<MstxRecord>();
    mstxRecord->markType = MarkType::RANGE_START_A;
    mstxRecord->rangeId = 8;
    instance.SetStepInfo(*mstxRecord);
    instance.config_.stepList.stepCount = 3;
    instance.config_.stepList.stepIdList[0] = 1;
    instance.config_.stepList.stepIdList[1] = 2;
    instance.config_.stepList.stepIdList[2] = 6;
    EXPECT_EQ(instance.IsNeedSkip(GD_INVALID_NUM), true);

    ResetEventReportStepInfo();
}

TEST(EventReportTest, TestReportSkipStepsWithMstxEndMismatch)
{
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectAllNpu = true;

    auto buffer = RecordBuffer::CreateRecordBuffer<MstxRecord>(TLVBlockType::MARK_MESSAGE, "step start");
    MstxRecord* mstxRecord = buffer.Cast<MstxRecord>();
    mstxRecord->markType = MarkType::RANGE_START_A;
    mstxRecord->rangeId = 8;
    instance.SetStepInfo(*mstxRecord);
    instance.config_.stepList.stepCount = 3;
    instance.config_.stepList.stepIdList[0] = 1;
    instance.config_.stepList.stepIdList[1] = 2;
    instance.config_.stepList.stepIdList[2] = 6;
    EXPECT_EQ(instance.IsNeedSkip(GD_INVALID_NUM), false);

    mstxRecord->markType = MarkType::RANGE_END;
    mstxRecord->rangeId = 9;
    instance.SetStepInfo(*mstxRecord);
    EXPECT_EQ(instance.IsNeedSkip(GD_INVALID_NUM), false);

    ResetEventReportStepInfo();
}

TEST(EventReportTest, TestReportSkipStepsWithOnlyMstxEnd)
{
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    instance.config_.collectAllNpu = true;
    
    instance.config_.stepList.stepCount = 3;
    instance.config_.stepList.stepIdList[0] = 1;
    instance.config_.stepList.stepIdList[1] = 2;
    instance.config_.stepList.stepIdList[2] = 6;

    MstxRecord mstxRecord = {};
    mstxRecord.markType = MarkType::RANGE_END;
    mstxRecord.rangeId = 9;
    instance.SetStepInfo(mstxRecord);
    EXPECT_EQ(instance.IsNeedSkip(GD_INVALID_NUM), true);

    ResetEventReportStepInfo();
}

TEST(EventReportTest, ReportTestWithNoReceiveServerInfo) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    uint64_t testAddr = 0x12345678;
    uint64_t testSize = 1024;
    unsigned long long flag = 0x1234;
    CallStackString callStack;
    instance.config_.collectCpu = true;
    EXPECT_TRUE(instance.ReportMalloc(testAddr, testSize, flag, callStack));
    EXPECT_TRUE(instance.ReportFree(testAddr, callStack));

    EXPECT_TRUE(instance.ReportHostMalloc(testAddr, testSize));
    EXPECT_TRUE(instance.ReportHostFree(testAddr));

    int16_t devId = 1;
    int16_t streamId = 1;
    int16_t taskId = 1;
    auto taskKey = std::make_tuple(devId, streamId, taskId);
    AclnnKernelMapInfo kernelLaunchInfo {};
    kernelLaunchInfo.taskKey = taskKey;
    kernelLaunchInfo.timestamp = 123;
    kernelLaunchInfo.kernelName = "add";
    EXPECT_TRUE(instance.ReportKernelLaunch(kernelLaunchInfo));

    RecordSubType subtype = {};
    EXPECT_TRUE(instance.ReportAclItf(subtype));

    auto memPoolRecord = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    EXPECT_TRUE(instance.ReportMemPoolRecord(memPoolRecord));

    auto mstxRecord = RecordBuffer::CreateRecordBuffer<MstxRecord>();
    EXPECT_TRUE(instance.ReportMark(mstxRecord));
}

constexpr uint64_t MEM_VIRT_BIT = 10;
constexpr uint64_t MEM_SVM_VAL = 0x0;
constexpr uint64_t MEM_DEV_VAL = 0x1;
constexpr uint64_t MEM_HOST_VAL = 0x2;
constexpr uint64_t MEM_DVPP_VAL = 0x3;
constexpr uint64_t MEM_INVALID_VAL = 0x4;

TEST(GetMemOpSpaceFuncTest, GetMemOpSpaceIfINvalid)
{
    unsigned long long flag = MEM_INVALID_VAL << MEM_VIRT_BIT;
    Leaks::MemOpSpace result = Leaks::GetMemOpSpace(flag);
    EXPECT_EQ(result, Leaks::MemOpSpace::INVALID);
}

TEST(GetMemOpSpaceFuncTest, GetMemOpSpaceIfSVM)
{
    unsigned long long flag = MEM_SVM_VAL << MEM_VIRT_BIT;
    Leaks::MemOpSpace result = Leaks::GetMemOpSpace(flag);
    EXPECT_EQ(result, Leaks::MemOpSpace::SVM);
}

TEST(GetMemOpSpaceFuncTest, GetMemOpSpaceIfDevice)
{
    unsigned long long flag = MEM_DEV_VAL << MEM_VIRT_BIT;
    Leaks::MemOpSpace result = Leaks::GetMemOpSpace(flag);
    EXPECT_EQ(result, Leaks::MemOpSpace::DEVICE);
}

TEST(GetMemOpSpaceFuncTest, GetMemOpSpaceIfHost)
{
    unsigned long long flag = MEM_HOST_VAL << MEM_VIRT_BIT;
    Leaks::MemOpSpace result = Leaks::GetMemOpSpace(flag);
    EXPECT_EQ(result, Leaks::MemOpSpace::HOST);
}

TEST(GetMemOpSpaceFuncTest, GetMemOpSpaceIfDvpp)
{
    unsigned long long flag = MEM_DVPP_VAL << MEM_VIRT_BIT;
    Leaks::MemOpSpace result = Leaks::GetMemOpSpace(flag);
    EXPECT_EQ(result, Leaks::MemOpSpace::DVPP);
}

TEST(GetMemOpSpaceFuncTest, GetMemOpSpaceIfOverType)
{
    unsigned long long flag = 0x5 << MEM_VIRT_BIT;
    Leaks::MemOpSpace result = Leaks::GetMemOpSpace(flag);
    EXPECT_EQ(result, Leaks::MemOpSpace::INVALID);
}

TEST(GetMemOpSpaceFuncTest, GetMemOpSpaceIfOverType1)
{
    unsigned long long flag = 0xF << MEM_VIRT_BIT;
    Leaks::MemOpSpace result = Leaks::GetMemOpSpace(flag);
    EXPECT_EQ(result, Leaks::MemOpSpace::INVALID);
}

AclItfRecord CreateAclItfRecord(RecordSubType type)
{
    auto record = AclItfRecord {};
    record.timestamp = Utility::GetTimeNanoseconds();
    record.subtype = type;
    record.pid = Utility::GetPid();
    record.tid = Utility::GetTid();
    return record;
}

KernelLaunchRecord CreateKernelLaunchRecord(KernelLaunchRecord kernelLaunchRecord)
{
    auto record = KernelLaunchRecord {};
    record = kernelLaunchRecord;
    record.timestamp = Utility::GetTimeNanoseconds();
    record.pid = Utility::GetPid();
    record.tid = Utility::GetTid();
    return record;
}

TEST(CreateAclItfRecordFuncTest, CreateAclItfRecordtestFinalize)
{
    Leaks::RecordSubType subtype = Leaks::RecordSubType::FINALIZE;
    Leaks::AclItfRecord record = CreateAclItfRecord(subtype);
    EXPECT_EQ(subtype, record.subtype);
}

TEST(CreateAclItfRecordFuncTest, CreateAclItfRecordtestINIT)
{
    Leaks::RecordSubType subtype = Leaks::RecordSubType::INIT;
    Leaks::AclItfRecord record = CreateAclItfRecord(subtype);
    EXPECT_EQ(subtype, record.subtype);
}

TEST(GetSpaceFunc, GetMemOpSpaceExpectSuccess)
{
    unsigned long long flag = 0b00100000000000;
    ASSERT_EQ(GetMemOpSpace(flag), MemOpSpace::HOST);
    flag = 0b00110000000000;
    ASSERT_EQ(GetMemOpSpace(flag), MemOpSpace::DVPP);
    flag = 0b11110000000000;
    ASSERT_EQ(GetMemOpSpace(flag), MemOpSpace::INVALID);
}