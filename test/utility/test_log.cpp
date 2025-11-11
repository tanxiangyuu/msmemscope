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
    LOG_DEBUG(testLog);
    logger.fp_ = nullptr;
    EXPECT_FALSE(FindStr("output.txt", testLog));
}

TEST(Log, log_warn_expect_output_warn_log)
{
    Log &logger = Log::GetLog();
    logger.fp_ = fopen("output.txt", "w");
    std::string testLog = "test log warn";
    Utility::SetLogLevel(Leaks::LogLv::WARN);
    LOG_WARN(testLog);
    logger.fp_ = nullptr;
    EXPECT_TRUE(FindStr("output.txt", testLog));
    EXPECT_TRUE(FindStr("output.txt", "[WARN]"));
}

TEST(Log, log_error_expect_output_erro_log)
{
    Log &logger = Log::GetLog();
    logger.fp_ = fopen("output.txt", "w");
    std::string testLog = "test log error";
    Utility::SetLogLevel(Leaks::LogLv::ERROR);
    LOG_ERROR(testLog);
    logger.fp_ = nullptr;
    EXPECT_TRUE(FindStr("output.txt", testLog));
    EXPECT_TRUE(FindStr("output.txt", "[ERROR]"));
}

TEST(Log, log_info_expect_output_info_log)
{
    Log &logger = Log::GetLog();
    logger.fp_ = fopen("output.txt", "w");
    std::string testLog = "test log info";
    Utility::SetLogLevel(Leaks::LogLv::INFO);
    LOG_INFO(testLog);
    logger.fp_ = nullptr;
    EXPECT_TRUE(FindStr("output.txt", testLog));
    EXPECT_TRUE(FindStr("output.txt", "[INFO]"));
}

TEST(Log, log_nullptr_expect_no_output)
{
    Log &logger = Log::GetLog();
    logger.fp_ = nullptr;

    logger.Printf("Test message", Leaks::LogLv::INFO, "test.cpp", 74);
    EXPECT_FALSE(FindStr("output.txt", "message"));
}

TEST(Log, log_level_different_threashold_expect_success)
{
    Log &logger = Log::GetLog();
    logger.fp_ = fopen("output.txt", "w");
    logger.SetLogLevel(Leaks::LogLv::INFO);

    logger.Printf("Debug message", Leaks::LogLv::DEBUG, "test.cpp", 84);
    logger.Printf("Info message", Leaks::LogLv::INFO, "test.cpp", 85);
    logger.Printf("Warning message", Leaks::LogLv::WARN, "test.cpp", 86);
    logger.Printf("Error message", Leaks::LogLv::ERROR, "test.cpp", 87);
    logger.fp_ = nullptr;
    EXPECT_TRUE(FindStr("output.txt", "message"));
}
