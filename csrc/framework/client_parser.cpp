// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "client_parser.h"

#include <unistd.h>
#include <iostream>
#include <sys/stat.h>
#include <getopt.h>
#include <regex>
#include "command.h"

#include "ustring.h"

namespace Leaks {

enum class OptVal : int32_t {
    SELECT_STEPS = 0,
    COMPARE,
    INPUT
};

void ShowDescription()
{
    std::cout <<
        "msleaks(MindStudio Leaks) is part of MindStudio Memory analysis Tools." << std::endl;
}

void ShowHelpInfo()
{
    ShowDescription();
    std::cout <<
        std::endl <<
        "Usage: msleaks <option(s)> prog-and-args" << std::endl <<
        std::endl <<
        "  basic user options, with default in [ ]:" << std::endl <<
        "    -h --help                      Show this message." << std::endl <<
        "    -v --version                   Show version." << std::endl <<
        "    -p --parse-kernel-name         Enable parse kernelLaunch name." << std::endl <<
        "    --steps=1,2,3,...              Select the steps to collect memory information." << std::endl <<
        "                                   The input step numbers need to be separated by, or ，." << std::endl <<
        "                                   The maximum number of steps is 5" << std::endl <<
        "    --compare                      Enable memory data comparison." << std::endl <<
        "    --input=path1,path2            paths to compare files, valid with compare command on" << std::endl <<
        "                                   The input paths need to be separated by, or ，." << std::endl;
}

void ShowVersion()
{
    ShowDescription();
    std::cout <<
        std::endl << "msleaks version " << "1.0" << std::endl;
}

bool UserCommandPrecheck(const UserCommand &userCommand)
{
    if (userCommand.config.enableCompare != userCommand.config.inputCorrectPaths) {
        std::cout << "Please use compare command with correct input paths!" << std::endl;
        return false;
    }

    return true;
}

void DoUserCommand(const UserCommand &userCommand)
{
    if (userCommand.printHelpInfo) {
        ShowHelpInfo();
        return ;
    }

    if (userCommand.printVersionInfo) {
        ShowVersion();
        return ;
    }

    if (!UserCommandPrecheck(userCommand)) {
        return;
    }

    Command command {userCommand};
    command.Exec();
}

void ClientParser::Interpretor(int32_t argc, char **argv)
{
    auto userCommand = Parse(argc, argv);
    DoUserCommand(userCommand);
}

std::vector<option> GetLongOptArray()
{
    std::vector<option> longOpts = {
        {"help", no_argument, nullptr, 'h'},
        {"version", no_argument, nullptr, 'v'},
        {"parse-kernel-name", no_argument, nullptr, 'p'},
        {"steps", required_argument, nullptr, static_cast<int32_t>(OptVal::SELECT_STEPS)},
        {"compare", no_argument, nullptr, static_cast<int32_t>(OptVal::COMPARE)},
        {"input", required_argument, nullptr, static_cast<int32_t>(OptVal::INPUT)},
        {nullptr, 0, nullptr, 0},
    };
    return longOpts;
}

std::string GetShortOptString(const std::vector<option> &longOptArray)
{
    // 根据long option string生成short option string
    std::string shortOpt;
    for (const auto &opt : longOptArray) {
        if (opt.name == nullptr) {
            break;
        }
        if ((opt.flag == nullptr) && (opt.val >= 'a') && (opt.val <= 'z')) {
            shortOpt.append(1, static_cast<char>(opt.val));
        }
    }
    return shortOpt;
}

static void ParseSelectSteps(const std::string &param, UserCommand &userCommand)
{
    std::regex dividePattern(R"([，,])");
    std::sregex_token_iterator  it(param.begin(), param.end(), dividePattern, -1);
    std::sregex_token_iterator  end;

    userCommand.config.stepList.stepCount = 0;
    std::regex numberPattern(R"(^[1-9]\d*$)");

    while (it != end) {
        SelectedStepList &stepListInfo = userCommand.config.stepList;

        if (stepListInfo.stepCount >= SELECTED_STEP_MAX_NUM) {
            break;
        }
        std::string step = it->str();
        if (!step.empty()) {
            if (!std::regex_match(step, numberPattern)) {
                std::cout << "[msleaks] ERROR: invalid steps input." << std::endl;
                userCommand.printHelpInfo = true;
                break;
            }
            stepListInfo.stepIdList[stepListInfo.stepCount] = stoi(it->str());
            stepListInfo.stepCount++;
        }
        
        it++;
    }

    return;
}

static void ParseInputPaths(const std::string param, UserCommand &userCommand)
{
    std::regex pattern(R"([，,])");
    std::sregex_token_iterator  it(param.begin(), param.end(), pattern, -1);
    std::sregex_token_iterator  end;

    while (it != end) {
        std::string path = it->str();
        if (!path.empty()) {
            userCommand.paths.emplace_back(path);
        }
        it++;
    }

    if (userCommand.paths.size() != PATHSIZE) {
        std::cout << "[msleaks] ERROR: invalid paths input." << std::endl;
        userCommand.printHelpInfo = true;
    } else {
        userCommand.config.inputCorrectPaths = true;
    }
}

void ParseUserCommand(const int32_t &opt, const std::string &param, UserCommand &userCommand)
{
    switch (opt) {
        case '?':
            std::cout << "[msleaks] ERROR: unrecognized command " << std::endl;
            userCommand.printHelpInfo = true;
            break;
        case 'h': // for --help
            userCommand.printHelpInfo = true;
            break;
        case 'v': // for --version
            userCommand.printVersionInfo = true;
            break;
        case 'p': // parse kernel name
            userCommand.config.parseKernelName = true;
            break;
        case static_cast<int32_t>(OptVal::SELECT_STEPS):
            ParseSelectSteps(param, userCommand);
            break;
        case static_cast<int32_t>(OptVal::COMPARE):
            userCommand.config.enableCompare = true;
            break;
        case static_cast<int32_t>(OptVal::INPUT):
            ParseInputPaths(param, userCommand);
            break;
        default:
            ;
    }
}

void ClientParser::InitialUserCommand(UserCommand &userCommand)
{
    userCommand.config.parseKernelName = false;
    userCommand.config.stepList.stepCount = 0;
    userCommand.config.enableCompare = false;
    userCommand.config.inputCorrectPaths = false;
}

UserCommand ClientParser::Parse(int32_t argc, char **argv)
{
    UserCommand userCommand;
    InitialUserCommand(userCommand);
    int32_t optionIndex = 0;
    int32_t opt = 0;
    auto longOptions = GetLongOptArray();
    std::string shortOptions = GetShortOptString(longOptions);
    optind = 0;
    while ((opt = getopt_long(argc, argv, shortOptions.c_str(), longOptions.data(),
        &optionIndex)) != -1) {
        // somehow optionIndex is not always correct for short option.
        // match it on our own.
        for (uint32_t i = 0; i < longOptions.size(); ++i) {
            if (longOptions[i].val == opt) {
                optionIndex = static_cast<int32_t>(i);
                break;
            }
        }
        std::string param;
        if (optarg) {
            param = optarg;
        }
        ParseUserCommand(opt, param, userCommand);
        // 打印help或者version不进行其他操作
        if (userCommand.printHelpInfo || userCommand.printVersionInfo) {
            return userCommand;
        }
    }
    std::vector<std::string> userBinCmd;
    for (; optind < argc; optind++) {
        userBinCmd.emplace_back(argv[optind]);
    }
    userCommand.cmd = userBinCmd;

    return userCommand;
}

}