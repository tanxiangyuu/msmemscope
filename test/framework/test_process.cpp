// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 
#include <gtest/gtest.h>
#include <vector>
#include "serializer.h"
#define private public
#include "process.h"
#undef private
#include "analysis/mstx_analyzer.h"
#include "analysis/dump_record.h"
#include "analysis/trace_record.h"
using namespace Leaks;

TEST(Process, process_launch_ls_expect_success)
{
    std::vector<std::string> execParams = {"/bin/ls"};
    std::stringstream buffer;
    std::streambuf *sbuf = std::cout.rdbuf();
    std::cout.rdbuf(buffer.rdbuf());
    std::string outputInfo = "user program exited";
    AnalysisConfig config;
    Process process(config);
    process.Launch(execParams);
    std::string captureInfo = buffer.str();
    EXPECT_EQ(captureInfo.find(outputInfo), std::string::npos);
    std::cout.rdbuf(sbuf);
}
 
TEST(Process, process_launch_empty_expect_success)
{
    std::vector<std::string> execParams = {""};
    std::stringstream buffer;
    std::streambuf *sbuf = std::cout.rdbuf();
    std::cout.rdbuf(buffer.rdbuf());
    std::string outputInfo = "exited abnormally";

    AnalysisConfig config;
    Process process(config);
    process.Launch(execParams);
    std::string captureInfo = buffer.str();
    EXPECT_NE(captureInfo.find(outputInfo), std::string::npos);
    std::cout.rdbuf(sbuf);
}
 
TEST(Process, process_setpreloadenv_expect_success)
{
    setenv("LD_PRELOAD_PATH", "/lib64/", 1);
    AnalysisConfig config;
    Process process(config);
    process.SetPreloadEnv();
    char *env = getenv("LD_PRELOAD");
    std::string hooksSo = "libascend_hal_hook.so:libhost_memory_hook.so:"
                          "libascend_mstx_hook.so:libascend_kernel_hook.so";
    EXPECT_EQ(std::string(env), hooksSo);
    setenv("LD_PRELOAD", "test.so", 1);
    process.SetPreloadEnv();
    env = getenv("LD_PRELOAD");
    EXPECT_EQ(std::string(env), hooksSo + ":test.so");
}
 
TEST(Process, process_postprocess_exit_signal_expect_success)
{
    std::vector<std::string> execParams = {"ls"};
    ExecCmd cmd(execParams);
    ::pid_t pid = ::fork();
    AnalysisConfig config;
    Process process(config);
    if (pid == 0) {
        sleep(200);
        _exit(EXIT_SUCCESS);
    } else {
        kill(pid, SIGTERM);

        std::stringstream buffer;
        std::streambuf *sbuf = std::cout.rdbuf();
        std::cout.rdbuf(buffer.rdbuf());
        std::string outputInfo = "user program exited by signal";
        process.PostProcess(cmd);
        std::string captureInfo = buffer.str();
        EXPECT_NE(captureInfo.find(outputInfo), std::string::npos);
        std::cout.rdbuf(sbuf);
    }
}

TEST(Process, process_postprocess_exit_abnormal_expect_success)
{
    std::vector<std::string> execParams = {""};
    ExecCmd cmd(execParams);
    ::pid_t pid = ::fork();
    AnalysisConfig config;
    Process process(config);
    if (pid == 0) {
        _exit(EXIT_FAILURE);
    } else {
        std::stringstream buffer;
        std::streambuf *sbuf = std::cout.rdbuf();
        std::cout.rdbuf(buffer.rdbuf());
        std::string outputInfo = "exited abnormally";
        process.PostProcess(cmd);
        std::string captureInfo = buffer.str();
        EXPECT_NE(captureInfo.find(outputInfo), std::string::npos);
        std::cout.rdbuf(sbuf);
    }
}

TEST(Process, do_dump_record_except_success)
{
    ClientId clientId = 0;
    EventRecord record{};

    DumpRecord::GetInstance().DumpData(clientId, record);
}

TEST(Process, do_record_handler_except_success)
{
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
    mstxRecordStart.rangeId = 0;
    mstxRecordStart.stepId = 1;
    mstxRecordStart.streamId = 123;
    record2.record.mstxRecord = mstxRecordStart;

    auto record3 = EventRecord{};
    record3.type = RecordType::KERNEL_LAUNCH_RECORD;
    auto kernelLaunchRecord = KernelLaunchRecord {};
    record3.record.kernelLaunchRecord = kernelLaunchRecord;

    auto record4 = EventRecord{};
    record4.type = RecordType::ACL_ITF_RECORD;
    auto aclItfRecord = AclItfRecord {};
    record4.record.aclItfRecord = aclItfRecord;
    RecordHandler(clientId, record1, analyzerfactory);
    RecordHandler(clientId, record2, analyzerfactory);
    RecordHandler(clientId, record3, analyzerfactory);
    RecordHandler(clientId, record4, analyzerfactory);
}

TEST(Process, do_msg_handler_record_packet_type_except_success)
{
    AnalysisConfig config;
    Process process(config);

    size_t clientId = 0;

    PacketHead head {PacketType::RECORD};
    auto record = EventRecord {};
    auto memRecord = MemOpRecord {};
    memRecord.recordIndex = 123;
    memRecord.kernelIndex = 123;
    memRecord.flag = 123;
    memRecord.pid = 123;
    memRecord.tid = 321;
    memRecord.devId = 9;
    memRecord.memType = MemOpType::MALLOC;
    memRecord.space = MemOpSpace::HOST;
    memRecord.modid = 234;
    memRecord.addr = 0x758;
    memRecord.memSize = 10240;
    memRecord.timeStamp = 1234567;
    record.type = RecordType::MEMORY_RECORD;
    record.record.memoryRecord = memRecord;
    std::string msg = Serialize(head, record);

    process.MsgHandle(clientId, msg);
}