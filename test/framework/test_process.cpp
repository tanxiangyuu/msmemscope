// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 
#include <gtest/gtest.h>
#include <vector>
#include "serializer.h"
#include "securec.h"
#define private public
#include "process.h"
#undef private
#include "analysis/mstx_analyzer.h"
#include "analysis/trace_record.h"
#include "client_process.h"
#include "client_parser.h"
#include "bit_field.h"
using namespace Leaks;

void setConfig(Config &config)
{
    BitField<decltype(config.eventType)> eventBit;
    BitField<decltype(config.levelType)> levelBit;
    BitField<decltype(config.analysisType)> analysisBit;
    analysisBit.setBit(static_cast<size_t>(AnalysisType::LEAKS_ANALYSIS));
    levelBit.setBit(static_cast<size_t>(LevelType::LEVEL_OP));
    levelBit.setBit(static_cast<size_t>(LevelType::LEVEL_KERNEL));
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::LAUNCH_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::ACCESS_EVENT));
    config.analysisType = analysisBit.getValue();
    config.eventType = eventBit.getValue();
    config.levelType = levelBit.getValue();
    config.enableCStack = true;
    config.enablePyStack = true;
    config.stepList.stepCount = 0;
    config.dataFormat = 0;
    strncpy_s(config.outputDir, sizeof(config.outputDir) - 1, "./testmsleaks", sizeof(config.outputDir) - 1);
}

TEST(Process, process_launch_ls_expect_success)
{
    std::vector<std::string> execParams = {"/bin/ls"};
    std::stringstream buffer;
    std::streambuf *sbuf = std::cout.rdbuf();
    std::cout.rdbuf(buffer.rdbuf());
    std::string outputInfo = "user program exited";
    Config config;
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

    Config config;
    Process process(config);
    process.Launch(execParams);
    std::string captureInfo = buffer.str();
    EXPECT_NE(captureInfo.find(outputInfo), std::string::npos);
    std::cout.rdbuf(sbuf);
}

TEST(Process, process_setpreloadenv_without_atb_expect_success)
{
    unsetenv("ATB_HOME_PATH");
    setenv("LD_PRELOAD_PATH", "/lib64/", 1);
    Config config;
    Process process(config);
    process.SetPreloadEnv();
    char *env = getenv("LD_PRELOAD");
    std::string hooksSo = "libleaks_ascend_hal_hook.so:libhost_memory_hook.so:"
                          "libascend_mstx_hook.so:libascend_kernel_hook.so";
    EXPECT_EQ(std::string(env), hooksSo);
    setenv("LD_PRELOAD", "test.so", 1);
    process.SetPreloadEnv();
    env = getenv("LD_PRELOAD");
    EXPECT_EQ(std::string(env), hooksSo + ":test.so");
    unsetenv("LD_PRELOAD");
}

TEST(Process, process_setpreloadenv_with_atb_abi_0_expect_success)
{
    setenv("ATB_HOME_PATH", "/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_0", 1);
    setenv("LD_PRELOAD_PATH", "/lib64/", 1);
    Config config;
    Process process(config);
    process.SetPreloadEnv();
    char *env = getenv("LD_PRELOAD");
    std::string hooksSo = "libleaks_ascend_hal_hook.so:libhost_memory_hook.so:"
                          "libascend_mstx_hook.so:libascend_kernel_hook.so:libatb_abi_0_hook.so";
    EXPECT_EQ(std::string(env), hooksSo);
    setenv("LD_PRELOAD", "test.so", 1);
    process.SetPreloadEnv();
    env = getenv("LD_PRELOAD");
    EXPECT_EQ(std::string(env), hooksSo + ":test.so");
    unsetenv("LD_PRELOAD");
}

TEST(Process, process_setpreloadenv_with_atb_abi_1_expect_success)
{
    setenv("ATB_HOME_PATH", "/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1", 1);
    setenv("LD_PRELOAD_PATH", "/lib64/", 1);
    Config config;
    Process process(config);
    process.SetPreloadEnv();
    char *env = getenv("LD_PRELOAD");
    std::string hooksSo = "libleaks_ascend_hal_hook.so:libhost_memory_hook.so:"
                          "libascend_mstx_hook.so:libascend_kernel_hook.so:libatb_abi_1_hook.so";
    EXPECT_EQ(std::string(env), hooksSo);
    setenv("LD_PRELOAD", "test.so", 1);
    process.SetPreloadEnv();
    env = getenv("LD_PRELOAD");
    EXPECT_EQ(std::string(env), hooksSo + ":test.so");
    unsetenv("LD_PRELOAD");
}
 
TEST(Process, process_postprocess_exit_signal_expect_success)
{
    std::vector<std::string> eEmptyParams;
    ExecCmd cmdEmpty(eEmptyParams);
    std::vector<std::string> execParams = {"ls"};
    ExecCmd cmd(execParams);
    ::pid_t pid = ::fork();
    Config config;
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
    Config config;
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

TEST(Process, do_record_handler_except_success)
{
    ClientId clientId = 0;
    auto buffer1 = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* record1 = buffer1.Cast<MemPoolRecord>();
    record1->type = RecordType::PTA_CACHING_POOL_RECORD;
    record1->recordIndex = 1;
    auto memoryusage1 = MemoryUsage {};
    memoryusage1.dataType = 0;
    memoryusage1.ptr = 12345;
    memoryusage1.allocSize = 512;
    memoryusage1.totalAllocated = 512;
    record1->memoryUsage = memoryusage1;

    auto buffer2 = RecordBuffer::CreateRecordBuffer<MstxRecord>();
    MstxRecord* record2 = buffer2.Cast<MstxRecord>();
    record2->type = RecordType::MSTX_MARK_RECORD;
    record2->markType = MarkType::RANGE_START_A;
    record2->rangeId = 0;
    record2->stepId = 1;
    record2->streamId = 123;

    auto buffer3 = RecordBuffer::CreateRecordBuffer<KernelLaunchRecord>();
    KernelLaunchRecord* record3 = buffer3.Cast<KernelLaunchRecord>();
    record3->type = RecordType::KERNEL_LAUNCH_RECORD;

    auto buffer4 = RecordBuffer::CreateRecordBuffer<AclItfRecord>();
    AclItfRecord* record4 = buffer4.Cast<AclItfRecord>();
    record4->type = RecordType::ACL_ITF_RECORD;

    auto buffer5 = RecordBuffer::CreateRecordBuffer<MemOpRecord>();
    MemOpRecord* record5 = buffer5.Cast<MemOpRecord>();
    record5->type = RecordType::MEMORY_RECORD;
    record5->subtype = RecordSubType::MALLOC;

    auto buffer6 = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* record6 = buffer6.Cast<MemPoolRecord>();
    record6->type = RecordType::PTA_WORKSPACE_POOL_RECORD;
    record6->recordIndex = 2;
    auto memoryusage6 = MemoryUsage {};
    memoryusage6.dataType = 0;
    memoryusage6.ptr = 12347;
    memoryusage6.allocSize = 512;
    memoryusage6.totalAllocated = 512;
    record6->memoryUsage = memoryusage6;

    Config config;
    setConfig(config);
    Process process(config);
    process.RecordHandler(clientId, buffer1);
    process.RecordHandler(clientId, buffer2);
    process.RecordHandler(clientId, buffer3);
    process.RecordHandler(clientId, buffer4);
    process.RecordHandler(clientId, buffer5);
    process.RecordHandler(clientId, buffer6);
}

TEST(Process, do_msg_handler_record_packet_type_except_success)
{
    Config config;
    setConfig(config);
    Process process(config);

    size_t clientId = 0;
    auto record = EventRecord {};
    auto memRecord = MemOpRecord {};
    memRecord.recordIndex = 123;
    memRecord.kernelIndex = 123;
    memRecord.flag = 123;
    memRecord.pid = 123;
    memRecord.tid = 321;
    memRecord.devId = 9;
    memRecord.subtype = RecordSubType::MALLOC;
    memRecord.space = MemOpSpace::HOST;
    memRecord.modid = 234;
    memRecord.addr = 0x758;
    memRecord.memSize = 10240;
    memRecord.timestamp = 1234567;
    record.type = RecordType::MEMORY_RECORD;
    record.record.memoryRecord = memRecord;
    std::string testMsg = "test";
    record.pyStackLen = testMsg.size();
    record.cStackLen = testMsg.size();
    PacketHead recordHead {PacketType::RECORD, sizeof(MemOpRecord) + record.pyStackLen + record.cStackLen};
    std::string str = Serialize(recordHead, record);
    str += testMsg + testMsg;
    process.MsgHandle(clientId, str);

    std::string logMsg = "test";
    PacketHead logHead {PacketType::LOG, logMsg.size()};
    std::string buffer = Serialize<PacketHead>(logHead);
    buffer += logMsg;
    process.MsgHandle(clientId, buffer);

    RecordBuffer rb = RecordBuffer::CreateRecordBuffer<MemOpRecord>(TLVBlockType::CALL_STACK_C, testMsg,
                                                                    TLVBlockType::CALL_STACK_PYTHON, testMsg);
    PacketHead newHead {PacketType::RECORD, rb.Size()};
    std::string buffer2 = Serialize<PacketHead>(newHead) + rb.Get();
    process.MsgHandle(clientId, buffer2);
}

TEST(Process, server_process_notify_test)
{
    std::string msg;
    ServerProcess server(LeaksCommType::SHARED_MEMORY);
    server.Start();
    server.Notify(0, msg);
}

TEST(Process, server_process_wait_test)
{
    std::string msg;
    ServerProcess server(LeaksCommType::SHARED_MEMORY);
    server.Start();
    server.Wait(0, msg);
}