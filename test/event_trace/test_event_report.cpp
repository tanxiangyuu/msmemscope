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
    EXPECT_TRUE(instance.ReportMalloc(testAddr, testSize, 1));
}

TEST(EventReportTest, ReportMallocTestHost) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    uint64_t testAddr = 0x12345678;
    uint64_t testSize = 1024;
    unsigned long long testFlag = 504403158274934784;
    MemOpSpace space = MemOpSpace::HOST;
    EXPECT_TRUE(instance.ReportMalloc(testAddr, testSize, 1));
}

TEST(EventReportTest, ReportFreeTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    uint64_t testAddr = 0x12345678;
    EXPECT_TRUE(instance.ReportFree(testAddr));
}

TEST(EventReportTest, ReportMarkTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    MstxRecord record;
    record.rangeId = 123;
    EXPECT_TRUE(instance.ReportMark(record));
}

TEST(EventReportTest, ReportKernelLaunchTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    KernelLaunchRecord record;
    void *hdl = nullptr;
    EXPECT_TRUE(instance.ReportKernelLaunch(record, hdl));
}

TEST(EventReportTest, ReportAclItfTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
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

TEST(EventReportTest, TestReportSkipStepsFuc)
{
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    instance.currentStep_ = 5;
    instance.config_.stepList.stepCount = 3;
    instance.config_.stepList.stepIdList[0] = 1;
    instance.config_.stepList.stepIdList[1] = 2;
    instance.config_.stepList.stepIdList[2] = 6;
    
    EXPECT_EQ(instance.IsNeedSkip(), true);

    instance.currentStep_ = 6;

    EXPECT_EQ(instance.IsNeedSkip(), false);
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