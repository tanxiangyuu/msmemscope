// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#define private public
#include "event_trace/trace_manager/event_trace_manager.h"
#include "event_trace/event_report.h"
#undef private
#include "event_trace/vallina_symbol.h"
#include "securec.h"
#include "bit_field.h"

using namespace MemScope;

class EventReportTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        Utility::FileCreateManager::GetInstance("./testmsmemscope").SetProjectDir("./testmsmemscope");
    }

    void TearDown() override
    {
        Utility::FileCreateManager::GetInstance("./testmsmemscope").SetProjectDir("");
        rmdir("./testmsmemscope");
    }
};

TEST_F(EventReportTest, EventReportInstanceTest)
{
    EventReport& instance1 = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
    EventReport& instance2 = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
    EXPECT_EQ(&instance1, &instance2);
}

TEST_F(EventReportTest, ReportMallocTestDEVICE)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);

    Config config = MemScope::GetConfig();
    config.collectAllNpu = true;
    BitField<decltype(config.eventType)> eventBit;
    BitField<decltype(config.levelType)> levelBit;
    levelBit.setBit(static_cast<size_t>(LevelType::LEVEL_OP));
    levelBit.setBit(static_cast<size_t>(LevelType::LEVEL_KERNEL));
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::LAUNCH_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::ACCESS_EVENT));
    config.eventType = eventBit.getValue();
    config.levelType = levelBit.getValue();
    config.enableCStack = true;
    config.enablePyStack = true;
    ConfigManager::Instance().SetConfig(config);
    uint64_t testAddr = 0x12345678;
    uint64_t testSize = 1024;
    unsigned long long testFlag = 2377900603261207558;
    MemOpSpace space = MemOpSpace::DEVICE;

    CallStackString callStack;
    EXPECT_TRUE(instance.ReportMalloc(testAddr, testSize, 1, callStack));
}

TEST_F(EventReportTest, ReportAddrInfoTest)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
    auto buffer = RecordBuffer::CreateRecordBuffer<AddrInfo>();
    EXPECT_TRUE(instance.ReportAddrInfo(buffer));
}

TEST_F(EventReportTest, ReportPyStepTest)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
    EXPECT_TRUE(instance.ReportPyStepRecord());
}

TEST_F(EventReportTest, ReportTorchNpuMallocTest)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
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
    Config config = MemScope::GetConfig();
    config.collectAllNpu = true;
    ConfigManager::Instance().SetConfig(config);
    EXPECT_TRUE(instance.ReportMemPoolRecord(buffer));
}

TEST_F(EventReportTest, ReportTorchNpuFreeTest)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
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
    Config config = MemScope::GetConfig();
    config.collectAllNpu = true;
    ConfigManager::Instance().SetConfig(config);
    EXPECT_TRUE(instance.ReportMemPoolRecord(buffer));
}

TEST_F(EventReportTest, ReportTorchNpuConditionTest)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
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

    EventTraceManager::Instance().SetTraceStatus(EventTraceStatus::NOT_IN_TRACING);
    EXPECT_TRUE(instance.ReportMemPoolRecord(buffer));
    EventTraceManager::Instance().SetTraceStatus(EventTraceStatus::IN_TRACING);
    Config config = MemScope::GetConfig();
    config.collectAllNpu = true;
    ConfigManager::Instance().SetConfig(config);
    EXPECT_TRUE(instance.ReportMemPoolRecord(buffer));
}

TEST_F(EventReportTest, ReportMallocTestHost)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
    Config config = MemScope::GetConfig();
    config.collectAllNpu = true;

    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::LAUNCH_EVENT));
    config.eventType = eventBit.getValue();
    config.enableCStack = true;
    config.enablePyStack = true;
    ConfigManager::Instance().SetConfig(config);
    uint64_t testAddr = 0x12345678;
    uint64_t testSize = 1024;
    unsigned long long testFlag = 504403158274934784;
    MemOpSpace space = MemOpSpace::HOST;
    CallStackString callStack;
    EXPECT_TRUE(instance.ReportMalloc(testAddr, testSize, 1, callStack));
}

TEST_F(EventReportTest, ReportFreeTest)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
    Config config = MemScope::GetConfig();
    config.collectAllNpu = true;
    BitField<decltype(config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::LAUNCH_EVENT));
    config.eventType = eventBit.getValue();
    config.enableCStack = true;
    config.enablePyStack = true;
    ConfigManager::Instance().SetConfig(config);
    uint64_t testAddr = 0x12345678;
    CallStackString callStack;
    EXPECT_TRUE(instance.ReportFree(testAddr, callStack));
}

TEST_F(EventReportTest, ReportMarkTest)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
    Config config = MemScope::GetConfig();
    config.collectAllNpu = true;
    ConfigManager::Instance().SetConfig(config);
    auto buffer = RecordBuffer::CreateRecordBuffer<MstxRecord>();
    MstxRecord* record = buffer.Cast<MstxRecord>();
    record->rangeId = 123;
    EXPECT_TRUE(instance.ReportMark(buffer));
}

TEST_F(EventReportTest, ReportKernelLaunchTest)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);

    Config config = MemScope::GetConfig();
    config.collectAllNpu = true;
    BitField<decltype(config.eventType)> eventBit;
    BitField<decltype(config.levelType)> levelBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::LAUNCH_EVENT));
    levelBit.setBit(static_cast<size_t>(LevelType::LEVEL_KERNEL));
    config.eventType = eventBit.getValue();
    ConfigManager::Instance().SetConfig(config);

    int16_t devId = 1;
    int16_t streamId = 1;
    int16_t taskId = 1;
    auto taskKey = std::make_tuple(devId, streamId, taskId);
    AclnnKernelMapInfo kernelLaunchInfo {};
    kernelLaunchInfo.taskKey = taskKey;
    kernelLaunchInfo.timestamp = 123;
    kernelLaunchInfo.kernelName = "add";
    EXPECT_TRUE(instance.ReportKernelLaunch(kernelLaunchInfo));

    int16_t devIdNeg = -1;
    auto taskKeyNeg = std::make_tuple(devIdNeg, streamId, taskId);
    AclnnKernelMapInfo kernelLaunchInfoNeg {};
    kernelLaunchInfoNeg.taskKey = taskKey;
    kernelLaunchInfoNeg.timestamp = 123;
    kernelLaunchInfoNeg.kernelName = "add";
    EXPECT_TRUE(instance.ReportKernelLaunch(kernelLaunchInfo));
}

TEST_F(EventReportTest, ReportAclItfTest)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);

    Config config = MemScope::GetConfig();
    config.collectAllNpu = true;
    ConfigManager::Instance().SetConfig(config);

    EXPECT_TRUE(instance.ReportAclItf(RecordSubType::INIT));
}

TEST_F(EventReportTest, ReportAtbOpExecuteTest)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
    Config config = MemScope::GetConfig();
    config.collectAllNpu = true;
    ConfigManager::Instance().SetConfig(config);
    char name[255] = "operation";
    char attr[255] = "path:model_struct";
    EXPECT_TRUE(instance.ReportAtbOpExecute(name, sizeof(name), attr, sizeof(attr), RecordSubType::ATB_START));
}

TEST_F(EventReportTest, ReportAtbKernelTest)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
    Config config = MemScope::GetConfig();
    config.collectAllNpu = true;
    ConfigManager::Instance().SetConfig(config);
    char name[255] = "operation";
    char attr[255] = "path:model_struct";
    EXPECT_TRUE(instance.ReportAtbKernel(name, sizeof(name), attr, sizeof(attr), RecordSubType::KERNEL_START));
}

TEST_F(EventReportTest, ReportAtbMemAccessTest)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
    Config config = MemScope::GetConfig();
    config.collectAllNpu = true;
    ConfigManager::Instance().SetConfig(config);
    char name[255] = "operation";
    char attr[255] = "path:model_struct";
    uint64_t addr = 123;
    uint64_t size = 1;
    EXPECT_TRUE(instance.ReportAtbAccessMemory(name, attr, addr, size, AccessType::UNKNOWN));
}

TEST_F(EventReportTest, ReportAtenLaunchTestExpectSuccess)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
    Config config = MemScope::GetConfig();
    config.collectAllNpu = true;
    ConfigManager::Instance().SetConfig(config);
    auto atenOpLaunchRecord = RecordBuffer::CreateRecordBuffer<AtenOpLaunchRecord>();
    EXPECT_TRUE(instance.ReportAtenLaunch(atenOpLaunchRecord));
}

TEST_F(EventReportTest, ReportAtenAccessTestExpectSuccess)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
    Config config = MemScope::GetConfig();
    config.collectAllNpu = true;
    ConfigManager::Instance().SetConfig(config);
    auto memAccessRecord = RecordBuffer::CreateRecordBuffer<MemAccessRecord>();
    EXPECT_TRUE(instance.ReportAtenAccess(memAccessRecord));
}

TEST_F(EventReportTest, ReportAtenLaunchTestExpextSuccess)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
    Config config = MemScope::GetConfig();
    config.collectAllNpu = true;
    ConfigManager::Instance().SetConfig(config);
    auto atenOpLaunchRecord = RecordBuffer::CreateRecordBuffer<AtenOpLaunchRecord>();
    EXPECT_TRUE(instance.ReportAtenLaunch(atenOpLaunchRecord));
}

TEST_F(EventReportTest, ReportAtenAccessTestExpextSuccess)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
    Config config = MemScope::GetConfig();
    config.collectAllNpu = true;
    ConfigManager::Instance().SetConfig(config);
    auto memAccessRecord = RecordBuffer::CreateRecordBuffer<MemAccessRecord>();
    EXPECT_TRUE(instance.ReportAtenAccess(memAccessRecord));
}

TEST_F(EventReportTest, ReportKernelExcuteTestExpextSuccess)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
    Config config = MemScope::GetConfig();
    config.collectAllNpu = true;
    ConfigManager::Instance().SetConfig(config);
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

TEST_F(EventReportTest, VallinaSymbolTest)
{
    using va = VallinaSymbol<TestLoader>;
    char const *symbol;
    va::Instance().Get(symbol);
}

void ResetEventReportStepInfo()
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
    instance.stepInfo_.currentStepId = 0;
    instance.stepInfo_.inStepRange = false;
    instance.stepInfo_.stepMarkRangeIdList.clear();
    return;
}

TEST_F(EventReportTest, TestReportSkipStepsNormal)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
    Config config = MemScope::GetConfig();
    config.collectAllNpu = true;
    config.stepList.stepCount = 3;
    config.stepList.stepIdList[0] = 1;
    config.stepList.stepIdList[1] = 2;
    config.stepList.stepIdList[2] = 6;
    ConfigManager::Instance().SetConfig(config);
    auto buffer = RecordBuffer::CreateRecordBuffer<MstxRecord>(TLVBlockType::MARK_MESSAGE, "step start");
    MstxRecord* mstxRecord = buffer.Cast<MstxRecord>();
    mstxRecord->markType = MarkType::RANGE_START_A;
    mstxRecord->rangeId = 8;
    instance.SetStepInfo(*mstxRecord);
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

TEST_F(EventReportTest, TestReportSkipStepsWithNoMstx)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
    Config config = MemScope::GetConfig();
    config.collectAllNpu = true;
    config.stepList.stepCount = 3;
    config.stepList.stepIdList[0] = 1;
    config.stepList.stepIdList[1] = 2;
    config.stepList.stepIdList[2] = 6;
    ConfigManager::Instance().SetConfig(config);

    EXPECT_EQ(instance.IsNeedSkip(GD_INVALID_NUM), true);

    ResetEventReportStepInfo();
}

TEST_F(EventReportTest, TestReportSkipStepsWithOtherMessageMstx)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
    Config config = MemScope::GetConfig();
    config.collectAllNpu = true;
    config.stepList.stepCount = 3;
    config.stepList.stepIdList[0] = 1;
    config.stepList.stepIdList[1] = 2;
    config.stepList.stepIdList[2] = 6;
    ConfigManager::Instance().SetConfig(config);
    auto buffer = RecordBuffer::CreateRecordBuffer<MstxRecord>(
        TLVBlockType::MARK_MESSAGE, "report host memory info start");
    MstxRecord* mstxRecord = buffer.Cast<MstxRecord>();
    mstxRecord->markType = MarkType::RANGE_START_A;
    mstxRecord->rangeId = 8;
    instance.SetStepInfo(*mstxRecord);
    EXPECT_EQ(instance.IsNeedSkip(GD_INVALID_NUM), true);

    ResetEventReportStepInfo();
}

TEST_F(EventReportTest, TestReportSkipStepsWithMstxEndMismatch)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
    Config config = MemScope::GetConfig();
    config.collectAllNpu = true;
    config.stepList.stepCount = 3;
    config.stepList.stepIdList[0] = 1;
    config.stepList.stepIdList[1] = 2;
    config.stepList.stepIdList[2] = 6;
    ConfigManager::Instance().SetConfig(config);

    auto buffer = RecordBuffer::CreateRecordBuffer<MstxRecord>(TLVBlockType::MARK_MESSAGE, "step start");
    MstxRecord* mstxRecord = buffer.Cast<MstxRecord>();
    mstxRecord->markType = MarkType::RANGE_START_A;
    mstxRecord->rangeId = 8;
    instance.SetStepInfo(*mstxRecord);
    EXPECT_EQ(instance.IsNeedSkip(GD_INVALID_NUM), false);

    mstxRecord->markType = MarkType::RANGE_END;
    mstxRecord->rangeId = 9;
    instance.SetStepInfo(*mstxRecord);
    EXPECT_EQ(instance.IsNeedSkip(GD_INVALID_NUM), false);

    ResetEventReportStepInfo();
}

TEST_F(EventReportTest, TestReportSkipStepsWithOnlyMstxEnd)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
    
    Config config = MemScope::GetConfig();
    config.collectAllNpu = true;
    config.stepList.stepCount = 3;
    config.stepList.stepIdList[0] = 1;
    config.stepList.stepIdList[1] = 2;
    config.stepList.stepIdList[2] = 6;
    ConfigManager::Instance().SetConfig(config);

    MstxRecord mstxRecord = {};
    mstxRecord.markType = MarkType::RANGE_END;
    mstxRecord.rangeId = 9;
    instance.SetStepInfo(mstxRecord);
    EXPECT_EQ(instance.IsNeedSkip(GD_INVALID_NUM), true);

    ResetEventReportStepInfo();
}

TEST_F(EventReportTest, ReportTestWithNoReceiveServerInfo)
{
    EventReport& instance = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
    uint64_t testAddr = 0x12345678;
    uint64_t testSize = 1024;
    unsigned long long flag = 0x1234;
    CallStackString callStack;
    Config config = MemScope::GetConfig();
    config.collectAllNpu = true;
    ConfigManager::Instance().SetConfig(config);

    EXPECT_TRUE(instance.ReportMalloc(testAddr, testSize, flag, callStack));
    EXPECT_TRUE(instance.ReportFree(testAddr, callStack));

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

TEST_F(EventReportTest, GetMemOpSpaceIfINvalid)
{
    unsigned long long flag = MEM_INVALID_VAL << MEM_VIRT_BIT;
    MemScope::MemOpSpace result = MemScope::GetMemOpSpace(flag);
    EXPECT_EQ(result, MemScope::MemOpSpace::INVALID);
}

TEST_F(EventReportTest, GetMemOpSpaceIfSVM)
{
    unsigned long long flag = MEM_SVM_VAL << MEM_VIRT_BIT;
    MemScope::MemOpSpace result = MemScope::GetMemOpSpace(flag);
    EXPECT_EQ(result, MemScope::MemOpSpace::SVM);
}

TEST_F(EventReportTest, GetMemOpSpaceIfDevice)
{
    unsigned long long flag = MEM_DEV_VAL << MEM_VIRT_BIT;
    MemScope::MemOpSpace result = MemScope::GetMemOpSpace(flag);
    EXPECT_EQ(result, MemScope::MemOpSpace::DEVICE);
}

TEST_F(EventReportTest, GetMemOpSpaceIfHost)
{
    unsigned long long flag = MEM_HOST_VAL << MEM_VIRT_BIT;
    MemScope::MemOpSpace result = MemScope::GetMemOpSpace(flag);
    EXPECT_EQ(result, MemScope::MemOpSpace::HOST);
}

TEST_F(EventReportTest, GetMemOpSpaceIfDvpp)
{
    unsigned long long flag = MEM_DVPP_VAL << MEM_VIRT_BIT;
    MemScope::MemOpSpace result = MemScope::GetMemOpSpace(flag);
    EXPECT_EQ(result, MemScope::MemOpSpace::DVPP);
}

TEST_F(EventReportTest, GetMemOpSpaceIfOverType)
{
    unsigned long long flag = 0x5 << MEM_VIRT_BIT;
    MemScope::MemOpSpace result = MemScope::GetMemOpSpace(flag);
    EXPECT_EQ(result, MemScope::MemOpSpace::INVALID);
}

TEST_F(EventReportTest, GetMemOpSpaceIfOverType1)
{
    unsigned long long flag = 0xF << MEM_VIRT_BIT;
    MemScope::MemOpSpace result = MemScope::GetMemOpSpace(flag);
    EXPECT_EQ(result, MemScope::MemOpSpace::INVALID);
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

TEST_F(EventReportTest, CreateAclItfRecordtestFinalize)
{
    MemScope::RecordSubType subtype = MemScope::RecordSubType::FINALIZE;
    MemScope::AclItfRecord record = CreateAclItfRecord(subtype);
    EXPECT_EQ(subtype, record.subtype);
}

TEST_F(EventReportTest, CreateAclItfRecordtestINIT)
{
    MemScope::RecordSubType subtype = MemScope::RecordSubType::INIT;
    MemScope::AclItfRecord record = CreateAclItfRecord(subtype);
    EXPECT_EQ(subtype, record.subtype);
}

TEST_F(EventReportTest, GetMemOpSpaceExpectSuccess)
{
    unsigned long long flag = 0b00100000000000;
    ASSERT_EQ(GetMemOpSpace(flag), MemOpSpace::HOST);
    flag = 0b00110000000000;
    ASSERT_EQ(GetMemOpSpace(flag), MemOpSpace::DVPP);
    flag = 0b11110000000000;
    ASSERT_EQ(GetMemOpSpace(flag), MemOpSpace::INVALID);
}