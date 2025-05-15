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
#include "client_process.h"
#include "client_parser.h"
#include "bit_field.h"
using namespace Leaks;

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

TEST(Process, do_dump_record_except_success)
{
    ClientId clientId = 0;
    Record record{};

    Config config;
    DumpRecord::GetInstance(config).DumpData(clientId, record);
}

TEST(Process, do_record_handler_except_success)
{
    ClientId clientId = 0;
    auto record1 = Record{};
    record1.eventRecord.type = RecordType::TORCH_NPU_RECORD;
    auto npuRecordMalloc = TorchNpuRecord {};
    npuRecordMalloc.recordIndex = 1;
    auto memoryusage1 = MemoryUsage {};
    memoryusage1.dataType = 0;
    memoryusage1.ptr = 12345;
    memoryusage1.allocSize = 512;
    memoryusage1.totalAllocated = 512;
    npuRecordMalloc.memoryUsage = memoryusage1;
    record1.eventRecord.record.torchNpuRecord = npuRecordMalloc;

    auto record2 = Record{};
    record2.eventRecord.type = RecordType::MSTX_MARK_RECORD;
    auto mstxRecordStart = MstxRecord {};
    mstxRecordStart.markType = MarkType::RANGE_START_A;
    mstxRecordStart.rangeId = 0;
    mstxRecordStart.stepId = 1;
    mstxRecordStart.streamId = 123;
    record2.eventRecord.record.mstxRecord = mstxRecordStart;

    auto record3 = Record{};
    record3.eventRecord.type = RecordType::KERNEL_LAUNCH_RECORD;
    auto kernelLaunchRecord = KernelLaunchRecord {};
    record3.eventRecord.record.kernelLaunchRecord = kernelLaunchRecord;

    auto record4 = Record{};
    record4.eventRecord.type = RecordType::ACL_ITF_RECORD;
    auto aclItfRecord = AclItfRecord {};
    record4.eventRecord.record.aclItfRecord = aclItfRecord;

    Config config;
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
    config.stepList.stepCount = 0;
    Process process(config);
    process.RecordHandler(clientId, record1);
    process.RecordHandler(clientId, record2);
    process.RecordHandler(clientId, record3);
    process.RecordHandler(clientId, record4);
}

TEST(Process, do_msg_handler_record_packet_type_except_success)
{
    Config config;
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
    config.stepList.stepCount = 0;
    Process process(config);

    size_t clientId = 0;

    PacketHead recordHead {PacketType::RECORD};
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
    std::string testMsg = "test";
    record.pyStackLen = testMsg.size();
    record.cStackLen = testMsg.size();
    std::string str = Serialize(recordHead, record);
    str += testMsg + testMsg;
    process.MsgHandle(clientId, str);

    std::string logMsg = "test";
    PacketHead logHead {PacketType::LOG};
    std::string buffer = Serialize<PacketHead, uint64_t>(logHead, logMsg.size());
    buffer += logMsg;
    process.MsgHandle(clientId, buffer);
}

TEST(Process, server_process_notify_test)
{
    std::string msg;
    ServerProcess server(CommType::SOCKET);
    server.Notify(0, msg);
}

TEST(Process, server_process_wait_test)
{
    std::string msg;
    ServerProcess server(CommType::SOCKET);
    server.Wait(0, msg);
}