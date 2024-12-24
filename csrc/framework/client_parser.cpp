// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "client_parser.h"

#include <unistd.h>
#include <iostream>
#include <sys/stat.h>
#include <getopt.h>
#include <regex>
#include "command.h"

namespace Leaks {

enum class OptVal : int32_t {
    SELECT_STEPS = 0,
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
        "    -h --help                              show this message" << std::endl <<
        "    -p --parse-kernel-name                 enable parse kernelLaunchName" << std::endl <<
        "    --select-steps=<step1,step2,...>       set select steps" << std::endl;
}

void DoUserCommand(const UserCommand &userCommand)
{
    if (userCommand.printHelpInfo) {
        ShowHelpInfo();
        return ;
    }
    Command command {userCommand.config};
    command.Exec(userCommand.cmd);
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
        {"parse-kernel-name", no_argument, nullptr, 'p'},
        {"select-steps", required_argument, nullptr, static_cast<int32_t>(OptVal::SELECT_STEPS)},
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
    std::regex pattern(R"(\d+)");
    std::sregex_iterator it(param.begin(), param.end(), pattern);
    std::sregex_iterator end;

    userCommand.config.stepList.stepCount = 0;

    while (it != end) {
        SelectedStepList &stepListInfo = userCommand.config.stepList;

        if (stepListInfo.stepCount >= SELECTED_STEP_MAX_NUM) {
            break;
        }

        stepListInfo.stepIdList[stepListInfo.stepCount] = stoi(it->str());
        stepListInfo.stepCount++;
        it++;
    }

    return;
}

void ParseUserCommand(const int32_t &opt, const std::string &param, UserCommand &userCommand)
{
    switch (opt) {
        case '?':
            std::cout << "[leaks] ERROR: unrecognized command " << std::endl;
            userCommand.printHelpInfo = true;
            break;
        case 'h': // for --help
            userCommand.printHelpInfo = true;
            break;
        case 'p': // parse kernel name
            userCommand.config.parseKernelName = true;
            break;
        case static_cast<int32_t>(OptVal::SELECT_STEPS):
            ParseSelectSteps(param, userCommand);
            break;
        default:
            ;
    }
}

void ClientParser::InitialUserCommand(UserCommand &userCommand)
{
    userCommand.config.parseKernelName = false;
    userCommand.config.stepList.stepCount = 0;
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
        // 打印help时不进行其他操作
        if (userCommand.printHelpInfo) {
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