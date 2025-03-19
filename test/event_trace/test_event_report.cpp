// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#define private public
#include "event_trace/event_report.h"
#undef private
#include "event_trace/vallina_symbol.h"
#include "handle_mapping.h"
#include "securec.h"

using namespace Leaks;

TEST(EventReportTest, EventReportInstanceTest) {
    EventReport& instance1 = EventReport::Instance(CommType::MEMORY);
    EventReport& instance2 = EventReport::Instance(CommType::MEMORY);
    EXPECT_EQ(&instance1, &instance2);
}

TEST(EventReportTest, ReportMallocTestDEVICE) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    uint64_t testAddr = 0x12345678;
    uint64_t testSize = 1024;
    unsigned long long testFlag = 2377900603261207558;
    MemOpSpace space = MemOpSpace::DEVICE;
    instance.isReceiveServerInfo_ = true;
    EXPECT_TRUE(instance.ReportMalloc(testAddr, testSize, 1));
}

TEST(EventReportTest, ReportMallocTestHost) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    uint64_t testAddr = 0x12345678;
    uint64_t testSize = 1024;
    unsigned long long testFlag = 504403158274934784;
    MemOpSpace space = MemOpSpace::HOST;
    instance.isReceiveServerInfo_ = true;
    EXPECT_TRUE(instance.ReportMalloc(testAddr, testSize, 1));
}

TEST(EventReportTest, ReportFreeTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    uint64_t testAddr = 0x12345678;
    instance.isReceiveServerInfo_ = true;
    EXPECT_TRUE(instance.ReportFree(testAddr));
}

TEST(EventReportTest, ReportHostMallocWithoutMstxTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    uint64_t testAddr = 0x12345678;
    uint64_t testSize = 1024;
    instance.isReceiveServerInfo_ = true;
    EXPECT_TRUE(instance.ReportHostMalloc(testAddr, testSize));
}
 
TEST(EventReportTest, ReportHostFreeWithoutMstxTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    uint64_t testAddr = 0x12345678;
    instance.isReceiveServerInfo_ = true;
    EXPECT_TRUE(instance.ReportHostFree(testAddr));
}

TEST(EventReportTest, ReportHostMallocTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    auto mstxRecordStart = MstxRecord {};
    mstxRecordStart.markType = MarkType::RANGE_START_A;
    mstxRecordStart.rangeId = 1;
    strncpy_s(mstxRecordStart.markMessage, sizeof(mstxRecordStart.markMessage),
        "report host memory info start", sizeof(mstxRecordStart.markMessage));
    instance.ReportMark(mstxRecordStart);

    uint64_t testAddr = 0x12345678;
    uint64_t testSize = 1024;
    EXPECT_TRUE(instance.ReportHostMalloc(testAddr, testSize));

    auto mstxRecordEnd = MstxRecord {};
    mstxRecordEnd.markType = MarkType::RANGE_END;
    mstxRecordEnd.rangeId = 1;
    instance.ReportMark(mstxRecordEnd);
}
 
TEST(EventReportTest, ReportHostFreeTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    auto mstxRecordStart = MstxRecord {};
    mstxRecordStart.markType = MarkType::RANGE_START_A;
    mstxRecordStart.rangeId = 1;
    strncpy_s(mstxRecordStart.markMessage, sizeof(mstxRecordStart.markMessage),
        "report host memory info start", sizeof(mstxRecordStart.markMessage));
    instance.ReportMark(mstxRecordStart);

    uint64_t testAddr = 0x12345678;
    EXPECT_TRUE(instance.ReportHostFree(testAddr));

    auto mstxRecordEnd = MstxRecord {};
    mstxRecordEnd.markType = MarkType::RANGE_END;
    mstxRecordEnd.rangeId = 1;
    instance.ReportMark(mstxRecordEnd);
}

TEST(EventReportTest, ReportMarkTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.isReceiveServerInfo_ = true;
    MstxRecord record;
    record.rangeId = 123;
    EXPECT_TRUE(instance.ReportMark(record));
}

TEST(EventReportTest, ReportKernelLaunchTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
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
    
    EXPECT_TRUE(instance.ReportMalloc(testAddr, testSize, flag));
    EXPECT_TRUE(instance.ReportFree(testAddr));

    EXPECT_TRUE(instance.ReportHostMalloc(testAddr, testSize));
    EXPECT_TRUE(instance.ReportHostFree(testAddr));

    KernelLaunchRecord kernelLaunchRecord = {};
    EXPECT_TRUE(instance.ReportKernelLaunch(kernelLaunchRecord, nullptr));

    AclOpType aclOpType = {};
    EXPECT_TRUE(instance.ReportAclItf(aclOpType));

    TorchNpuRecord torchNpuRecord = {};
    EXPECT_TRUE(instance.ReportTorchNpu(torchNpuRecord));

    MstxRecord mstxRecord = {};
    EXPECT_TRUE(instance.ReportMark(mstxRecord));
}

TEST(KernelNameFunc, PipeCallGivenLsCommandReturnFalse)
{
    std::vector<std::string> argv = {"/bin/ls", "/tmp"};
    std::string output;
    ASSERT_TRUE(PipeCall(argv, output));
    ASSERT_FALSE(output.empty());
}

TEST(KernelNameFunc, PipeCallGivenTestCommandReturnFalse)
{
    std::vector<std::string> argv = {"test-command"};
    std::string output;
    ASSERT_FALSE(PipeCall(argv, output));
}

TEST(KernelNameFunc, WriteBinaryGivenValidDataReturnSuccess)
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

TEST(KernelNameFunc, ParseLineGivenValidKernelNameLineReturnTrueName)
{
    std::string line = "0000 g F .text ReduceSum_7a09f_high_performance_123_mix_aiv";
    std::string kernelName;
    kernelName = ParseLine(line);
    ASSERT_EQ(kernelName, "ReduceSum_7a09f");
}

TEST(KernelNameFunc, ParseLineGivenInvalidKernelNameLineReturnEmptyName)
{
    std::string line = "000 g O .data g_opSystemRunCfg";
    std::string kernelName;
    kernelName = ParseLine(line);
    ASSERT_EQ(kernelName, "");
}

TEST(KernelNameFunc, ParseNameFromOutputGivenValidSymbolTableReturnTrueName)
{
    std::string output = ("SYMBOL TABLE:\n"
    "000 g F .text test_000_mix_aic"
    "000 g O .data g_opSystemRunCfg\n");
    std::string kernelName;
    kernelName = ParseNameFromOutput(output);
    ASSERT_EQ(kernelName, "test_000");
}

TEST(KernelNameFunc, ParseNameFromOutputGivenInValidSymbolTableReturnEmptyName)
{
    std::string output = ("TEST TABLE:\n"
    "000 g F .text test_000_mix_aic"
    "000 g O .data g_opSystemRunCfg\n");
    std::string kernelName;
    kernelName = ParseNameFromOutput(output);
    ASSERT_EQ(kernelName, "");
}

TEST(KernelNameFunc, GetNameFromBinaryGivenHdlReturnEmptyName)
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