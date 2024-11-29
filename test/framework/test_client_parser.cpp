// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <string>
#include <vector>

#define private public
#include "client_parser.h"
#undef private

using namespace Leaks;

void InvalidParamCheckHelpInfo(const char* invalidInput)
{
    std::vector<const char*> argv = {
        "leaks",
        invalidInput
    };
 
    optind = 1;
    ClientParser cliParser;
    testing::internal::CaptureStdout();
    cliParser.Interpretor(argv.size(), const_cast<char**>(argv.data()));
    std::string capture = testing::internal::GetCapturedStdout();
    ASSERT_NE(capture.find("Usage"), std::string::npos);
}
 
TEST(ClientParser, pass_help_parameter_expect_get_print_help_info)
{
    std::vector<const char*> argv = {
        "leaks",
        "--help"
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printHelpInfo);
}
 
TEST(ClientParser, pass_empty_prog_name_expect_get_empty_bin_cmd)
{
    std::vector<const char*> argv = {
        "leaks"
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.cmd.empty());
}
 
TEST(ClientParser, pass_test_as_prog_expect_get_bin_cmd_test)
{
    std::vector<const char*> argv = {
        "leaks",
        "test"
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_FALSE(cmd.cmd.empty());
    ASSERT_EQ(cmd.cmd[0], "test");
}
 
TEST(ClientParser, invalid_single_dash_option_expect_one_error)
{
    std::vector<const char*> argv = {
        "leaks",
        "-log-file=log.txt"
    };
 
    ClientParser cliParser;
    testing::internal::CaptureStdout();
    cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    std::string capture = testing::internal::GetCapturedStdout();
    const char *errorStr = "unrecognized command";
    size_t pos = capture.find(errorStr);
    ASSERT_NE(pos, std::string::npos) << capture;
    pos = capture.find(errorStr, pos + 1);
    ASSERT_EQ(pos, std::string::npos);
}