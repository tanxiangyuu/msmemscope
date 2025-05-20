// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#define private public
#include "event_trace/event_report.h"
#undef private
#include "event_trace/vallina_symbol.h"
#include "handle_mapping.h"
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
    CallStackString callStack;
    EXPECT_TRUE(instance.ReportMalloc(testAddr, testSize, 1, callStack));
}

TEST(EventReportTest, ReportTorchNpuMallocTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);

    auto npuRecordMalloc = MemPoolRecord {};
    npuRecordMalloc.recordIndex = 1;
    auto memoryusage1 = MemoryUsage {};
    memoryusage1.deviceIndex = 0;
    memoryusage1.dataType = 0;
    memoryusage1.ptr = 12345;
    memoryusage1.allocSize = 512;
    memoryusage1.totalAllocated = 512;
    npuRecordMalloc.memoryUsage = memoryusage1;
    instance.isReceiveServerInfo_ = true;
    CallStackString callStack;
    EXPECT_TRUE(instance.ReportTorchNpu(npuRecordMalloc, callStack));
}

TEST(EventReportTest, ReportTorchNpuFreeTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);

    auto npuRecordFree = MemPoolRecord {};
    npuRecordFree.recordIndex = 3;
    auto memoryusage1 = MemoryUsage {};
    memoryusage1.deviceIndex = 3;
    memoryusage1.dataType = 1;
    memoryusage1.ptr = 12345;
    memoryusage1.allocSize = 512;
    memoryusage1.totalAllocated = 512;
    npuRecordFree.memoryUsage = memoryusage1;
    instance.isReceiveServerInfo_ = true;
    CallStackString callStack;
    EXPECT_TRUE(instance.ReportTorchNpu(npuRecordFree, callStack));
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
    auto mstxRecordStart = MstxRecord{};
    CallStackString stack;
    mstxRecordStart.markType = MarkType::RANGE_START_A;
    mstxRecordStart.rangeId = 1;
    strncpy_s(mstxRecordStart.markMessage, sizeof(mstxRecordStart.markMessage),
        "report host memory info start", sizeof(mstxRecordStart.markMessage));
    instance.ReportMark(mstxRecordStart, stack);

    uint64_t testAddr = 0x12345678;
    uint64_t testSize = 1024;
    CallStackString callStack;
    EXPECT_TRUE(instance.ReportHostMalloc(testAddr, testSize));

    auto mstxRecordEnd = MstxRecord {};
    mstxRecordEnd.markType = MarkType::RANGE_END;
    mstxRecordEnd.rangeId = 1;
    instance.ReportMark(mstxRecordEnd, stack);
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
    auto mstxRecordStart = MstxRecord {};
    mstxRecordStart.markType = MarkType::RANGE_START_A;
    mstxRecordStart.rangeId = 1;
    strncpy_s(mstxRecordStart.markMessage, sizeof(mstxRecordStart.markMessage),
        "report host memory info start", sizeof(mstxRecordStart.markMessage));
    instance.ReportMark(mstxRecordStart, callStack);
    uint64_t testAddr = 0x12345678;
    EXPECT_TRUE(instance.ReportHostFree(testAddr));

    auto mstxRecordEnd = MstxRecord {};
    mstxRecordEnd.markType = MarkType::RANGE_END;
    mstxRecordEnd.rangeId = 1;
    instance.ReportMark(mstxRecordEnd, callStack);
}

TEST(EventReportTest, ReportMarkTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    MstxRecord record;
    record.rangeId = 123;
    CallStackString callStack;
    EXPECT_TRUE(instance.ReportMark(record, callStack));
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
    KernelLaunchRecord record;
    void *hdl = nullptr;
    EXPECT_TRUE(instance.ReportKernelLaunch(record, hdl));
}

TEST(EventReportTest, ReportAclItfTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    EXPECT_TRUE(instance.ReportAclItf(AclOpType::INIT));
}

TEST(EventReportTest, ReportAtbOpExecuteTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    AtbOpExecuteRecord atbOpExecuteRecord;
    EXPECT_TRUE(instance.ReportAtbOpExecute(atbOpExecuteRecord));
}

TEST(EventReportTest, ReportAtbKernelTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    AtbKernelRecord atbKernelRecord;
    EXPECT_TRUE(instance.ReportAtbKernel(atbKernelRecord));
}

TEST(EventReportTest, ReportAtbMemAccessTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    MemAccessRecord memAccessRecord;
    std::vector<MemAccessRecord> records;
    records.push_back(memAccessRecord);
    EXPECT_TRUE(instance.ReportAtbAccessMemory(records));
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
    MstxRecord mstxRecord = {};
    mstxRecord.markType = MarkType::RANGE_START_A;
    strncpy_s(mstxRecord.markMessage, sizeof(mstxRecord.markMessage),
        "step start", sizeof(mstxRecord.markMessage));
    mstxRecord.rangeId = 8;
    instance.SetStepInfo(mstxRecord);
    instance.config_.stepList.stepCount = 3;
    instance.config_.stepList.stepIdList[0] = 1;
    instance.config_.stepList.stepIdList[1] = 2;
    instance.config_.stepList.stepIdList[2] = 6;
    EXPECT_EQ(instance.IsNeedSkip(), false);

    mstxRecord.markType = MarkType::RANGE_END;
    instance.SetStepInfo(mstxRecord);
    EXPECT_EQ(instance.IsNeedSkip(), true);

    mstxRecord.markType = MarkType::RANGE_START_A;
    mstxRecord.rangeId = 9;
    instance.SetStepInfo(mstxRecord);
    EXPECT_EQ(instance.IsNeedSkip(), false);

    mstxRecord.markType = MarkType::RANGE_END;
    instance.SetStepInfo(mstxRecord);
    EXPECT_EQ(instance.IsNeedSkip(), true);

    mstxRecord.markType = MarkType::RANGE_START_A;
    mstxRecord.rangeId = 10;
    instance.SetStepInfo(mstxRecord);
    EXPECT_EQ(instance.IsNeedSkip(), true);

    mstxRecord.markType = MarkType::RANGE_END;
    instance.SetStepInfo(mstxRecord);
    EXPECT_EQ(instance.IsNeedSkip(), true);

    ResetEventReportStepInfo();
}

TEST(EventReportTest, TestReportSkipStepsWithNoMstx)
{
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    instance.config_.stepList.stepCount = 3;
    instance.config_.stepList.stepIdList[0] = 1;
    instance.config_.stepList.stepIdList[1] = 2;
    instance.config_.stepList.stepIdList[2] = 6;
    EXPECT_EQ(instance.IsNeedSkip(), true);

    ResetEventReportStepInfo();
}

TEST(EventReportTest, TestReportSkipStepsWithOtherMessageMstx)
{
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    MstxRecord mstxRecord = {};
    mstxRecord.markType = MarkType::RANGE_START_A;
    strncpy_s(mstxRecord.markMessage, sizeof(mstxRecord.markMessage),
        "report host memory info start", sizeof(mstxRecord.markMessage));
    mstxRecord.rangeId = 8;
    instance.SetStepInfo(mstxRecord);
    instance.config_.stepList.stepCount = 3;
    instance.config_.stepList.stepIdList[0] = 1;
    instance.config_.stepList.stepIdList[1] = 2;
    instance.config_.stepList.stepIdList[2] = 6;
    EXPECT_EQ(instance.IsNeedSkip(), true);

    ResetEventReportStepInfo();
}

TEST(EventReportTest, TestReportSkipStepsWithMstxEndMismatch)
{
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    MstxRecord mstxRecord = {};
    mstxRecord.markType = MarkType::RANGE_START_A;
    strncpy_s(mstxRecord.markMessage, sizeof(mstxRecord.markMessage),
        "step start", sizeof(mstxRecord.markMessage));
    mstxRecord.rangeId = 8;
    instance.SetStepInfo(mstxRecord);
    instance.config_.stepList.stepCount = 3;
    instance.config_.stepList.stepIdList[0] = 1;
    instance.config_.stepList.stepIdList[1] = 2;
    instance.config_.stepList.stepIdList[2] = 6;
    EXPECT_EQ(instance.IsNeedSkip(), false);

    mstxRecord.markType = MarkType::RANGE_END;
    mstxRecord.rangeId = 9;
    instance.SetStepInfo(mstxRecord);
    EXPECT_EQ(instance.IsNeedSkip(), false);

    ResetEventReportStepInfo();
}

TEST(EventReportTest, TestReportSkipStepsWithOnlyMstxEnd)
{
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    
    instance.config_.stepList.stepCount = 3;
    instance.config_.stepList.stepIdList[0] = 1;
    instance.config_.stepList.stepIdList[1] = 2;
    instance.config_.stepList.stepIdList[2] = 6;

    MstxRecord mstxRecord = {};
    mstxRecord.markType = MarkType::RANGE_END;
    mstxRecord.rangeId = 9;
    instance.SetStepInfo(mstxRecord);
    EXPECT_EQ(instance.IsNeedSkip(), true);

    ResetEventReportStepInfo();
}

TEST(EventReportTest, ReportTestWithNoReceiveServerInfo) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    uint64_t testAddr = 0x12345678;
    uint64_t testSize = 1024;
    unsigned long long flag = 0x1234;
    CallStackString callStack;
    EXPECT_TRUE(instance.ReportMalloc(testAddr, testSize, flag, callStack));
    EXPECT_TRUE(instance.ReportFree(testAddr, callStack));

    EXPECT_TRUE(instance.ReportHostMalloc(testAddr, testSize));
    EXPECT_TRUE(instance.ReportHostFree(testAddr));

    KernelLaunchRecord kernelLaunchRecord = {};
    EXPECT_TRUE(instance.ReportKernelLaunch(kernelLaunchRecord, nullptr));

    AclOpType aclOpType = {};
    EXPECT_TRUE(instance.ReportAclItf(aclOpType));

    MemPoolRecord memPoolRecord = {};
    EXPECT_TRUE(instance.ReportTorchNpu(memPoolRecord, callStack));

    MstxRecord mstxRecord = {};
    EXPECT_TRUE(instance.ReportMark(mstxRecord, callStack));
}

TEST(KernelNameFuncTest, PipeCallGivenLsCommandReturnFalse)
{
    std::vector<std::string> argv = {"/bin/ls", "/tmp"};
    std::string output;
    ASSERT_TRUE(PipeCall(argv, output));
    ASSERT_FALSE(output.empty());
}

TEST(KernelNameFuncTest, PipeCallGivenTestCommandReturnFalse)
{
    std::vector<std::string> argv = {"test-command"};
    std::string output;
    ASSERT_FALSE(PipeCall(argv, output));
}

TEST(KernelNameFuncTest, WriteBinaryGivenValidDataReturnSuccess)
{
    std::string fileName = "./test.bin";
    char wbuf[] = "123456789";
    ASSERT_TRUE(WriteBinary(fileName, wbuf, sizeof(wbuf)));

    std::ifstream ifs;
    ifs.open(fileName, std::ios::binary);
    ASSERT_TRUE(ifs.is_open());
    char rbuf[sizeof(wbuf)];
    ifs.read(rbuf, sizeof(rbuf));
    ifs.close();
    ASSERT_EQ(std::string(wbuf), std::string(rbuf));

    std::remove(fileName.c_str());
}

TEST(KernelNameFuncTest, ParseLineGivenValidKernelNameLineReturnTrueName)
{
    std::string line = "0000 g F .text ReduceSum_7a09f_high_performance_123_mix_aiv";
    std::string kernelName;
    kernelName = ParseLine(line);
    ASSERT_EQ(kernelName, "ReduceSum_7a09f");
}

TEST(KernelNameFuncTest, ParseLineGivenInvalidKernelNameLineReturnEmptyName)
{
    std::string line = "000 g O .data g_opSystemRunCfg";
    std::string kernelName;
    kernelName = ParseLine(line);
    ASSERT_EQ(kernelName, "");
}

TEST(KernelNameFuncTest, ParseNameFromOutputGivenValidSymbolTableReturnTrueName)
{
    std::string output = ("SYMBOL TABLE:\n"
    "000 g F .text test_000_mix_aic"
    "000 g O .data g_opSystemRunCfg\n");
    std::string kernelName;
    kernelName = ParseNameFromOutput(output);
    ASSERT_EQ(kernelName, "test_000");
}

TEST(KernelNameFuncTest, ParseNameFromOutputGivenInValidSymbolTableReturnEmptyName)
{
    std::string output = ("TEST TABLE:\n"
    "000 g F .text test_000_mix_aic"
    "000 g O .data g_opSystemRunCfg\n");
    std::string kernelName;
    kernelName = ParseNameFromOutput(output);
    ASSERT_EQ(kernelName, "");
}

TEST(KernelNameFuncTest, GetNameFromBinaryGivenHdlReturnEmptyName)
{
    std::vector<uint8_t> handleData{1, 2, 3};
    void *hdl = handleData.data();
    Leaks::BinKernel binData {};
    binData.bin = {0x01, 0x02, 0x03, 0x04};
    std::string kernelName;
    Leaks::HandleMapping::GetInstance().handleBinKernelMap_.insert({hdl, binData});
    kernelName = GetNameFromBinary(hdl);
    Leaks::HandleMapping::GetInstance().handleBinKernelMap_.erase(hdl);
    ASSERT_EQ(kernelName, "");
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

MemOpRecord CreateMemRecord(MemOpType type, unsigned long long flag, MemOpSpace space, uint64_t addr, uint64_t size)
{
    MemOpRecord record;
    record.timeStamp = Utility::GetTimeMicroseconds();
    record.flag = flag;
    record.memType = type;
    record.space = space;
    record.addr = addr;
    record.memSize = size;
    record.pid = Utility::GetPid();
    record.tid = Utility::GetTid();
    return record;
}

AclItfRecord CreateAclItfRecord(AclOpType type)
{
    auto record = AclItfRecord {};
    record.timeStamp = Utility::GetTimeMicroseconds();
    record.type = type;
    record.pid = Utility::GetPid();
    record.tid = Utility::GetTid();
    return record;
}

KernelLaunchRecord CreateKernelLaunchRecord(KernelLaunchRecord kernelLaunchRecord)
{
    auto record = KernelLaunchRecord {};
    record = kernelLaunchRecord;
    record.timeStamp = Utility::GetTimeMicroseconds();
    record.pid = Utility::GetPid();
    record.tid = Utility::GetTid();
    return record;
}

TEST(CreateMemRecordFuncTest, CreateMemRecordtestMalloc)
{
    Leaks::MemOpType type = Leaks::MemOpType::MALLOC;
    unsigned long long flag = 2377900603261207558;
    Leaks::MemOpSpace space = Leaks::MemOpSpace::SVM;
    uint64_t addr = 0x12345678;
    uint64_t size = 1024;
    Leaks::MemOpRecord record = CreateMemRecord(type, flag, space, addr, size);
    EXPECT_EQ(type, record.memType);
    EXPECT_EQ(flag, record.flag);
    EXPECT_EQ(space, record.space);
    EXPECT_EQ(addr, record.addr);
    EXPECT_EQ(size, record.memSize);
}

TEST(CreateMemRecordFuncTest, CreateMemRecordtestFree)
{
    Leaks::MemOpType type = Leaks::MemOpType::FREE;
    unsigned long long flag = 2377900603261207558;
    Leaks::MemOpSpace space = Leaks::MemOpSpace::SVM;
    uint64_t addr = 0x12345678;
    uint64_t size = 1024;
    Leaks::MemOpRecord record = CreateMemRecord(type, flag, space, addr, size);
    EXPECT_EQ(type, record.memType);
    EXPECT_EQ(flag, record.flag);
    EXPECT_EQ(space, record.space);
    EXPECT_EQ(addr, record.addr);
    EXPECT_EQ(size, record.memSize);
}

TEST(CreateMemRecordFuncTest, CreateMemRecordtestDevice)
{
    Leaks::MemOpType type = Leaks::MemOpType::FREE;
    unsigned long long flag = 2377900603261207558;
    Leaks::MemOpSpace space = Leaks::MemOpSpace::DEVICE;
    uint64_t addr = 0x12345678;
    uint64_t size = 1024;
    Leaks::MemOpRecord record = CreateMemRecord(type, flag, space, addr, size);
    EXPECT_EQ(type, record.memType);
    EXPECT_EQ(flag, record.flag);
    EXPECT_EQ(space, record.space);
    EXPECT_EQ(addr, record.addr);
    EXPECT_EQ(size, record.memSize);
}

TEST(CreateMemRecordFuncTest, CreateMemRecordtestHost)
{
    Leaks::MemOpType type = Leaks::MemOpType::FREE;
    unsigned long long flag = 2377900603261207558;
    Leaks::MemOpSpace space = Leaks::MemOpSpace::HOST;
    uint64_t addr = 0x12345678;
    uint64_t size = 1024;
    Leaks::MemOpRecord record = CreateMemRecord(type, flag, space, addr, size);
    EXPECT_EQ(type, record.memType);
    EXPECT_EQ(flag, record.flag);
    EXPECT_EQ(space, record.space);
    EXPECT_EQ(addr, record.addr);
    EXPECT_EQ(size, record.memSize);
}

TEST(CreateMemRecordFuncTest, CreateMemRecordtestDVPP)
{
    Leaks::MemOpType type = Leaks::MemOpType::FREE;
    unsigned long long flag = 2377900603261207558;
    Leaks::MemOpSpace space = Leaks::MemOpSpace::DVPP;
    uint64_t addr = 0x12345678;
    uint64_t size = 1024;
    Leaks::MemOpRecord record = CreateMemRecord(type, flag, space, addr, size);
    EXPECT_EQ(type, record.memType);
    EXPECT_EQ(flag, record.flag);
    EXPECT_EQ(space, record.space);
    EXPECT_EQ(addr, record.addr);
    EXPECT_EQ(size, record.memSize);
}

TEST(CreateAclItfRecordFuncTest, CreateAclItfRecordtestFinalize)
{
    Leaks::AclOpType aclOpType = Leaks::AclOpType::FINALIZE;
    Leaks::AclItfRecord record = CreateAclItfRecord(aclOpType);
    EXPECT_EQ(aclOpType, record.type);
}

TEST(CreateAclItfRecordFuncTest, CreateAclItfRecordtestINIT)
{
    Leaks::AclOpType aclOpType = Leaks::AclOpType::INIT;
    Leaks::AclItfRecord record = CreateAclItfRecord(aclOpType);
    EXPECT_EQ(aclOpType, record.type);
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

TEST(EventReportTest, ToRawCArgvExpectSuccess)
{
    std::vector<std::string> argv = {"test"};
    ToRawCArgv(argv);
}