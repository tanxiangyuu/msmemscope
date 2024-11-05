// Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 
#include <gtest/gtest.h>
#include <vector>

#define private public
#include "process.h"
#undef private

using namespace Leaks;

TEST(Process, process_launch_ls_expect_success)
{
    std::vector<std::string> execParams = {"/bin/ls"};
    std::stringstream buffer;
    std::streambuf *sbuf = std::cout.rdbuf();
    std::cout.rdbuf(buffer.rdbuf());
    std::string outputInfo = "user program exited";

    Process process;
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
    std::string outputInfo = "user program exited abnormally";

    Process process;
    process.Launch(execParams);
    std::string captureInfo = buffer.str();
    EXPECT_NE(captureInfo.find(outputInfo), std::string::npos);
    std::cout.rdbuf(sbuf);
}
 
TEST(Process, process_setpreloadenv_expect_success)
{
    setenv("LD_PRELOAD_PATH", "/lib64/", 1);
    Process process;
    process.SetPreloadEnv();
    char* env = getenv("LD_PRELOAD");
    ASSERT_STREQ(env, "libascend_hal_hook.so");
    setenv("LD_PRELOAD", "test.so", 1);
    process.SetPreloadEnv();
    env = getenv("LD_PRELOAD");
    ASSERT_STREQ(env, "libascend_hal_hook.so:test.so");
}
 
TEST(Process, process_postprocess_exit_signal_expect_success)
{
    ::pid_t pid = ::fork();
    Process process;
    if (pid == 0) {
        sleep(200);
        _exit(EXIT_SUCCESS);
    } else {
        kill(pid, SIGTERM);

        std::stringstream buffer;
        std::streambuf *sbuf = std::cout.rdbuf();
        std::cout.rdbuf(buffer.rdbuf());
        std::string outputInfo = "user program exited by signal";
        process.PostProcess();
        std::string captureInfo = buffer.str();
        EXPECT_NE(captureInfo.find(outputInfo), std::string::npos);
        std::cout.rdbuf(sbuf);
    }
}

TEST(Process, process_postprocess_exit_abnormal_expect_success)
{
    ::pid_t pid = ::fork();
    Process process;
    if (pid == 0) {
        _exit(EXIT_FAILURE);
    } else {
        std::stringstream buffer;
        std::streambuf *sbuf = std::cout.rdbuf();
        std::cout.rdbuf(buffer.rdbuf());
        std::string outputInfo = "user program exited abnormally";
        process.PostProcess();
        std::string captureInfo = buffer.str();
        EXPECT_NE(captureInfo.find(outputInfo), std::string::npos);
        std::cout.rdbuf(sbuf);
    }
}