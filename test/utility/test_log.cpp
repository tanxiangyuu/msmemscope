// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>

#define private public
#include "utility/log.h"
#undef private

using namespace Utility;

bool FindStr(const std::string &fileName, const std::string &str)
{
    FILE *fp = fopen(fileName.c_str(), "r");
    char buffer[256];
    if (fgets(buffer, sizeof(buffer), fp) != NULL && strstr(buffer, str.c_str()) != nullptr) {
        fclose(fp);
        return true;
    }
    fclose(fp);
    return false;
}

TEST(Log, log_debug_with_default_log_level_warn_expect_not_output)
{
    Log &logger = Log::GetLog();
    logger.fp_ = fopen("output.txt", "w");
    std::string testLog = "test log debug";
    LogDebug(testLog);
    logger.fp_ = nullptr;
    EXPECT_FALSE(FindStr("output.txt", testLog));
}

TEST(Log, log_warn_expect_output_warn_log)
{
    Log &logger = Log::GetLog();
    logger.fp_ = fopen("output.txt", "w");
    std::string testLog = "test log warn";
    Utility::SetLogLevel(Utility::LogLv::WARN);
    LogWarn(testLog);
    logger.fp_ = nullptr;
    EXPECT_TRUE(FindStr("output.txt", testLog));
    EXPECT_TRUE(FindStr("output.txt", "[WARN]"));
}

TEST(Log, log_error_expect_output_erro_log)
{
    Log &logger = Log::GetLog();
    logger.fp_ = fopen("output.txt", "w");
    std::string testLog = "test log error";
    Utility::SetLogLevel(Utility::LogLv::ERROR);
    LogError(testLog);
    logger.fp_ = nullptr;
    EXPECT_TRUE(FindStr("output.txt", testLog));
    EXPECT_TRUE(FindStr("output.txt", "[ERROR]"));
}

TEST(Log, log_info_expect_output_info_log)
{
    Log &logger = Log::GetLog();
    logger.fp_ = fopen("output.txt", "w");
    std::string testLog = "test log info";
    Utility::SetLogLevel(Utility::LogLv::INFO);
    LogInfo(testLog);
    logger.fp_ = nullptr;
    EXPECT_TRUE(FindStr("output.txt", testLog));
    EXPECT_TRUE(FindStr("output.txt", "[INFO]"));
}

TEST(Log, log_nullptr_expect_no_output)
{
    Log &logger = Log::GetLog();
    logger.fp_ = nullptr;

    logger.Printf("Test message", LogLv::INFO);
    EXPECT_FALSE(FindStr("output.txt", "message"));
}

TEST(Log, log_level_different_threashold_expect_success)
{
    Log &logger = Log::GetLog();
    logger.fp_ = fopen("output.txt", "w");
    logger.SetLogLevel(Utility::LogLv::INFO);

    logger.Printf("Debug message", LogLv::DEBUG);
    logger.Printf("Info message", LogLv::INFO);
    logger.Printf("Warning message", LogLv::WARN);
    logger.Printf("Error message", LogLv::ERROR);
    logger.fp_ = nullptr;
    EXPECT_TRUE(FindStr("output.txt", "message"));
}

TEST(Log, log_recv_expect_success)
{
    Log &logger = Log::GetLog();
    logger.fp_ = fopen("output.txt", "w");
    std::string testLog = "test log debug";
    LogRecv(testLog);
    logger.fp_ = nullptr;
    EXPECT_TRUE(FindStr("output.txt", testLog));
}