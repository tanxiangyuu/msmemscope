// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>

#define private public
#include "utility/log.h"
#undef private

using namespace Utility;

TEST(Log, log_debug_with_default_log_level_warn_expect_not_output)
{
    testing::internal::CaptureStdout();
    std::string testLog = "test log debug";
    LogDebug(testLog);
    std::string captureLog = testing::internal::GetCapturedStdout();
    ASSERT_EQ(captureLog.find(testLog), std::string::npos);
}

TEST(Log, log_warn_expect_output_warn_log)
{
    testing::internal::CaptureStdout();
    std::string testLog = "test log warn";
    LogWarn(testLog);
    std::string captureLog = testing::internal::GetCapturedStdout();
    ASSERT_NE(captureLog.find("[WARN]"), std::string::npos);
    ASSERT_NE(captureLog.find(testLog), std::string::npos);
}

TEST(Log, log_error_expect_output_erro_log)
{
    testing::internal::CaptureStdout();
    std::string testLog = "test log error";
    LogError(testLog);
    std::string captureLog = testing::internal::GetCapturedStdout();
    ASSERT_NE(captureLog.find("[ERROR]"), std::string::npos);
    ASSERT_NE(captureLog.find(testLog), std::string::npos);
}

TEST(Log, log_info_expect_output_info_log)
{
    testing::internal::CaptureStdout();
    std::string testLog = "test log info";
    LogInfo(testLog);
    std::string captureLog = testing::internal::GetCapturedStdout();
    ASSERT_NE(captureLog.find("[INFO]"), std::string::npos);
    ASSERT_NE(captureLog.find(testLog), std::string::npos);
}

TEST(Log, log_wrong_level_expect_output_info_log)
{
    testing::internal::CaptureStdout();
    std::string logLevel = "5";
    SetLogLevel(logLevel);
    std::string warnInfo = "LOG_LEVEL can only be set 0,1,2,3";
    std::string captureLog = testing::internal::GetCapturedStdout();
    ASSERT_NE(captureLog.find("[WARN]"), std::string::npos);
    ASSERT_NE(captureLog.find(warnInfo), std::string::npos);
}

TEST(Log, log_true_level_expect_output_info_log)
{
    testing::internal::CaptureStdout();
    std::string logLevel = "2";
    SetLogLevel(logLevel);
    std::string warnInfo = "LOG_LEVEL can only be set 0,1,2,3";
    std::string captureLog = testing::internal::GetCapturedStdout();
    ASSERT_EQ(captureLog.find("[WARN]"), std::string::npos);
    ASSERT_EQ(captureLog.find(warnInfo), std::string::npos);
}

TEST(Log, log_nullptr_expect_no_output)
{
    testing::internal::CaptureStdout();
    Log &logger = Log::GetLog();
    logger.fp_ = nullptr;

    logger.Printf("Test message", LogLv::INFO);
    std::string captureLog = testing::internal::GetCapturedStdout();
    ASSERT_EQ(captureLog.find("message"), std::string::npos);
}

TEST(Log, log_level_different_threashold_expect_success)
{
    testing::internal::CaptureStdout();
    Log &logger = Log::GetLog();
    logger.SetLogLevel("2");
    
    logger.Printf("Debug message", LogLv::DEBUG);
    logger.Printf("Info message", LogLv::INFO);
    logger.Printf("Warning message", LogLv::WARN);
    logger.Printf("Error message", LogLv::ERROR);
    std::string captureLog = testing::internal::GetCapturedStdout();
    ASSERT_EQ(captureLog.find("message"), std::string::npos);
}