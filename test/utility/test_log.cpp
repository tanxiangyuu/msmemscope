// Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>

#include "utility/log.h"

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