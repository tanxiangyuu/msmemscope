// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <string>
#include <vector>

#define private public
#include "client_parser.h"
#include "log.h"
#undef private

using namespace MemScope;

void InvalidParamCheckHelpInfo(const char* invalidInput)
{
    std::vector<const char*> argv = {
        "msmemscope",
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
        "msmemscope",
        "--help"
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printHelpInfo);
}

TEST(ClientParser, pass_help_parameter_expect_show_help_info)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--help"
    };

    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    testing::internal::CaptureStdout();
    cliParser.Interpretor(argv.size(), const_cast<char**>(argv.data()));
    std::string capture = testing::internal::GetCapturedStdout();
    ASSERT_NE(capture.find("Usage"), std::string::npos);
}

TEST(ClientParser, pass_valid_analysis_type_case_not_set)
{
    std::vector<const char*> argv = {
        "msmemscope",
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_EQ(cmd.config.analysisType, 1);
}

TEST(ClientParser, pass_valid_analysis_type_case_memscope)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--analysis=memscope"
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_EQ(cmd.config.analysisType, 1);
}

TEST(ClientParser, pass_valid_analysis_type_case_decompose)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--analysis=decompose"
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_EQ(cmd.config.analysisType, 2);
}

TEST(ClientParser, pass_valid_analysis_type_case_all)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--analysis=memscope,decompose"
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_EQ(cmd.config.eventType, 7);
}

TEST(ClientParser, pass_invalid_analysis_type_case)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--analysis=lekas"
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printHelpInfo);
}

TEST(ClientParser, pass_valid_level_value_expect_level0)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--level=0"
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_EQ(cmd.config.levelType, 1);
}

TEST(ClientParser, pass_valid_level_value_expect_level1)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--level=1"
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_EQ(cmd.config.levelType, 2);
}

TEST(ClientParser, pass_valid_level_value_expect_level_0_and_level_1)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--level=0,1"
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_EQ(cmd.config.levelType, 3);
}

TEST(ClientParser, pass_invalid_level_expect_show_help_info)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--level=3"
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printHelpInfo);
}

TEST(ClientParser, pass_valid_event_type_case_not_set)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--level=3"
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_EQ(cmd.config.eventType, 7);
}

TEST(ClientParser, pass_valid_event_type_case_alloc)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--events=alloc"
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_EQ(cmd.config.eventType, 1);
}

TEST(ClientParser, pass_valid_event_type_case_free)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--events=free"
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_EQ(cmd.config.eventType, 2);
}

TEST(ClientParser, pass_valid_event_type_case_launch)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--events=launch"
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_EQ(cmd.config.eventType, 4);
}

TEST(ClientParser, pass_valid_event_type_case_access)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--events=access"
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_EQ(cmd.config.eventType, 8);
}

TEST(ClientParser, pass_valid_event_type_case_all)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--events=alloc,free,launch,access"
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_EQ(cmd.config.eventType, 15);
}

TEST(ClientParser, pass_invalid_event_type_case)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--events=alloc,free,launhc"
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
        "msmemscope"
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
        "msmemscope",
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
        "msmemscope",
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

TEST(ClientParser, test_parse_select_steps)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--steps=2,3,123"
    };

    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_EQ(cmd.config.stepList.stepCount, 3);
    ASSERT_EQ(cmd.config.stepList.stepIdList[0], 2);
    ASSERT_EQ(cmd.config.stepList.stepIdList[1], 3);
    ASSERT_EQ(cmd.config.stepList.stepIdList[2], 123);

    argv = {
        "msmemscope",
        "--steps=2，3,234"
    };

    cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_EQ(cmd.config.stepList.stepCount, 3);
    ASSERT_EQ(cmd.config.stepList.stepIdList[0], 2);
    ASSERT_EQ(cmd.config.stepList.stepIdList[1], 3);
    ASSERT_EQ(cmd.config.stepList.stepIdList[2], 234);

    argv = {
        "msmemscope",
        "--steps=4294967295"
    };

    cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_EQ(cmd.config.stepList.stepCount, 1);
    ASSERT_EQ(cmd.config.stepList.stepIdList[0], 4294967295);
}

TEST(ClientParser, test_invalid_select_steps)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--steps=-1,0,3"
    };

    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printHelpInfo);

    argv = {
        "msmemscope",
        "--steps=2:3.4"
    };

    cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printHelpInfo);

    argv = {
        "msmemscope",
        "--steps=4294967296"
    };

    cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printHelpInfo);

    argv = {
        "msmemscope",
        "--steps=429496729500"
    };

    cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printHelpInfo);
}

TEST(ClientParser, test_exceed_five_steps_expect_print_help_info)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--steps=1,2,3,4,5,6"
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printHelpInfo);
}

TEST(ClientParser, test_compare_dump_data)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--compare",
        "--input=path1,path2"
    };

    // Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.config.enableCompare);
    ASSERT_FALSE(cmd.config.inputCorrectPaths);

    argv = {
        "msmemscope",
        "--compare",
        "--input=path1，path2"
    };

    // Reset getopt states
    cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.config.enableCompare);
    ASSERT_FALSE(cmd.config.inputCorrectPaths);
}

TEST(ClientParser, test_invalid_compare_dump_data)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--input=path1"
    };

    // Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_FALSE(cmd.config.enableCompare);
    ASSERT_FALSE(cmd.config.inputCorrectPaths);

    argv = {
        "msmemscope",
        "--compare"
    };

    // Reset getopt states
    cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.config.enableCompare);
    ASSERT_FALSE(cmd.config.inputCorrectPaths);
}

TEST(ClientParser, test_watch_config_set_all)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--watch=start:123,end,full-content"
    };

    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_EQ(std::string(cmd.config.watchConfig.start), "start");
    ASSERT_EQ(std::string(cmd.config.watchConfig.end), "end");
    ASSERT_EQ(cmd.config.watchConfig.outputId, 123);
    ASSERT_EQ(cmd.config.watchConfig.fullContent, true);
    ASSERT_EQ(cmd.config.watchConfig.isWatched, true);
}

TEST(ClientParser, test_watch_config_set_only_end)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--watch=,end,"
    };

    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_EQ(std::string(cmd.config.watchConfig.start), "");
    ASSERT_EQ(std::string(cmd.config.watchConfig.end), "end");
    ASSERT_EQ(cmd.config.watchConfig.outputId, UINT32_MAX);
    ASSERT_EQ(cmd.config.watchConfig.fullContent, false);
    ASSERT_EQ(cmd.config.watchConfig.isWatched, true);
}

TEST(ClientParser, test_watch_config_set_only_start_and_end)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--watch=start,end,"
    };

    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_EQ(std::string(cmd.config.watchConfig.start), "start");
    ASSERT_EQ(std::string(cmd.config.watchConfig.end), "end");
    ASSERT_EQ(cmd.config.watchConfig.outputId, UINT32_MAX);
    ASSERT_EQ(cmd.config.watchConfig.fullContent, false);
    ASSERT_EQ(cmd.config.watchConfig.isWatched, true);
}

TEST(ClientParser, test_watch_config_set_only_start_with_id_and_end)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--watch=start:123,end,"
    };

    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_EQ(std::string(cmd.config.watchConfig.start), "start");
    ASSERT_EQ(std::string(cmd.config.watchConfig.end), "end");
    ASSERT_EQ(cmd.config.watchConfig.outputId, 123);
    ASSERT_EQ(cmd.config.watchConfig.fullContent, false);
    ASSERT_EQ(cmd.config.watchConfig.isWatched, true);
}

TEST(ClientParser, test_watch_config_unset)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--level=0"
    };

    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_EQ(std::string(cmd.config.watchConfig.start), "");
    ASSERT_EQ(std::string(cmd.config.watchConfig.end), "");
    ASSERT_EQ(cmd.config.watchConfig.outputId, UINT32_MAX);
    ASSERT_EQ(cmd.config.watchConfig.fullContent, false);
    ASSERT_EQ(cmd.config.watchConfig.isWatched, false);
}

TEST(ClientParser, test_watch_config_null)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--watch="
    };

    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printHelpInfo);
}

TEST(ClientParser, test_watch_config_empty_end)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--watch=,"
    };

    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printHelpInfo);
}

TEST(ClientParser, test_watch_config_error_outputid)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--watch=start:error,end"
    };

    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printHelpInfo);
}

TEST(ClientParser, test_watch_config_error_start_part)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--watch=:,end"
    };

    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printHelpInfo);
}

TEST(ClientParser, test_watch_config_error_full_content)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--watch=start:123,end,full-contents"
    };

    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printHelpInfo);
}

TEST(ClientParser, test_collect_mode_full)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--collect-mode=immediate"
    };

    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_FALSE(cmd.printHelpInfo);
    ASSERT_EQ(cmd.config.collectMode, static_cast<std::uint8_t>(CollectMode::IMMEDIATE));
}

TEST(ClientParser, test_collect_mode_custom)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--collect-mode=deferred"
    };

    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_FALSE(cmd.printHelpInfo);
    ASSERT_EQ(cmd.config.collectMode, static_cast<std::uint8_t>(CollectMode::DEFERRED));
}

TEST(ClientParser, test_collect_mode_empty)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--collect-mode="
    };

    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printHelpInfo);
    ASSERT_EQ(cmd.config.collectMode, static_cast<std::uint8_t>(CollectMode::IMMEDIATE));
}

TEST(ClientParser, test_collect_mode_input_error)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--collect-mode=full2"
    };

    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printHelpInfo);
    ASSERT_EQ(cmd.config.collectMode, static_cast<std::uint8_t>(CollectMode::IMMEDIATE));
}

TEST(ClientParser, test_print_version)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--version"
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printVersionInfo);
}

TEST(ClientParser, print_version)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--version"
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    testing::internal::CaptureStdout();
    cliParser.Interpretor(argv.size(), const_cast<char**>(argv.data()));
    std::string capture = testing::internal::GetCapturedStdout();
    ASSERT_EQ(capture.find("Usage"), std::string::npos);
}

TEST(ClientParser, test_not_set_output)
{
    std::vector<const char*> argv = {
        "msmemscope",
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_EQ(cmd.outputPath, "");
}

TEST(ClientParser, test_set_invalid_output)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--output= "
    };

    // Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.outputPath.empty());

    argv = {
        "msmemscope",
        "--output=/MyPath1/MyPath2?"
    };
    cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.outputPath.empty());

    argv = {
        "msmemscope",
        "--output=/MyPath1/MyPath2*/"
    };
    cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.outputPath.empty());

    argv = {
        "msmemscope",
        "--output=MyPath|"
    };
    cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.outputPath.empty());

    std::string str(256, 'A');
    str = "--output=" + str;
    argv = {
        "msmemscope",
        str.c_str()
    };
    cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.outputPath.empty());
}

TEST(ClientParser, test_input_valid_log_level_expect_valid_loglv)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--log-level=warn"
    };
 
    /// Reset getopt states
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    Utility::SetLogLevel(static_cast<LogLv>(cmd.config.logLevel));
    ASSERT_EQ(Utility::Log::GetLog().lv_, LogLv::WARN);

    argv = {
        "msmemscope",
        "--log-level=info"
    };
 
    cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    Utility::SetLogLevel(static_cast<LogLv>(cmd.config.logLevel));
    ASSERT_EQ(Utility::Log::GetLog().lv_, LogLv::INFO);

    argv = {
        "msmemscope",
        "--log-level=error"
    };
 
    cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    Utility::SetLogLevel(static_cast<LogLv>(cmd.config.logLevel));
    ASSERT_EQ(Utility::Log::GetLog().lv_, LogLv::ERROR);
}

TEST(ClientParser, test_input_invalid_log_level_expect_invalid_loglv)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--log-level=test"
    };
 
    /// Reset getopt states
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printHelpInfo);

    argv = {
        "msmemscope",
        "--log-level="
    };
 
    cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printHelpInfo);
}

TEST(ClientParser, test_parse_call_stack_expect_true)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--call-stack=c:10,python:10"
    };
    /// Reset getopt states
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    argv = {
        "msmemscope",
        "--call-stack=c:00"
    };
    cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printHelpInfo);
    argv = {
        "msmemscope",
        "--call-stack="
    };
    cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_FALSE(cmd.printHelpInfo);
    argv = {
        "msmemscope",
        "--call-stack=c:10,c:15"
    };
    cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_FALSE(cmd.printHelpInfo);
    ASSERT_EQ(cmd.config.enableCStack, true);
    ASSERT_EQ(cmd.config.cStackDepth, 15);
    argv = {
        "msmemscope",
        "--call-stack=c:13,python:10,python:12"
    };
    cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_FALSE(cmd.printHelpInfo);
    ASSERT_EQ(cmd.config.enableCStack, true);
    ASSERT_EQ(cmd.config.cStackDepth, 13);
    ASSERT_EQ(cmd.config.enablePyStack, true);
    ASSERT_EQ(cmd.config.pyStackDepth, 12);
}

TEST(ClientParser, pass_data_format_case_db)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--data-format=db",
        "--output=./testmsmemscope"
    };
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_EQ(cmd.config.dataFormat, 1);
}

TEST(ClientParser, usercommand_precheck_false)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--compare"
    };
    optind = 1;
    ClientParser cliParser;
    testing::internal::CaptureStdout();
    cliParser.Interpretor(argv.size(), const_cast<char**>(argv.data()));
    std::string capture = testing::internal::GetCapturedStdout();
    ASSERT_NE(capture.find("Usage"), std::string::npos);

    argv = {
        "msmemscope",
        "--output="
    };
    testing::internal::CaptureStdout();
    cliParser.Interpretor(argv.size(), const_cast<char**>(argv.data()));
    capture = testing::internal::GetCapturedStdout();
    ASSERT_NE(capture.find("Usage"), std::string::npos);
}

TEST(ClientParser, test_invalid_device_case)
{
    std::vector<const char*> argv = {
        "msmemscope",
        "--device="
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printHelpInfo);

    argv = {
        "msmemscope",
        "--device=test"
    };
 
    /// Reset getopt states
    optind = 1;
    cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printHelpInfo);

    argv = {
        "msmemscope",
        "--device=npu:100"
    };
 
    /// Reset getopt states
    optind = 1;
    cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printHelpInfo);

    argv = {
        "msmemscope",
        "--device=npu:xx"
    };
 
    /// Reset getopt states
    optind = 1;
    cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.printHelpInfo);
}


TEST(ClientParser, test_valid_device_case)
{
    std::vector<const char*> argv = {
        "msmemscope",
    };
 
    /// Reset getopt states
    optind = 1;
    ClientParser cliParser;
    UserCommand cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.config.collectAllNpu);

    argv = {
        "msmemscope",
        "--device=npu"
    };
 
    /// Reset getopt states
    optind = 1;
    cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_TRUE(cmd.config.collectAllNpu);

    argv = {
        "msmemscope",
        "--device=npu:0,npu:2"
    };
 
    /// Reset getopt states
    optind = 1;
    cmd = cliParser.Parse(argv.size(), const_cast<char**>(argv.data()));
    ASSERT_FALSE(cmd.config.collectAllNpu);
    ASSERT_EQ(cmd.config.npuSlots, 5);
}