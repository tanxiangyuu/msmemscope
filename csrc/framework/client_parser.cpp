// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "client_parser.h"
#include <unistd.h>
#include <iostream>
#include <vector>
#include <sys/stat.h>
#include <getopt.h>
#include "command.h"

namespace Leaks {

void ClientParser::Interpretor(int32_t argc, char **argv)
{
    auto userCommand = Parse(argc, argv);
    Command command {userCommand.config};
    command.Exec(userCommand.cmd);
    return;
}

std::vector<option> GetLongOptArray()
{
    std::vector<option> longOpts = {
        {"help", no_argument, nullptr, 'h'},
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
            if (opt.has_arg == optional_argument) {
                shortOpt.append(2, ':'); // 可不跟参数使用2个冒号，如 "a::"
            } else if (opt.has_arg == required_argument) {
                shortOpt.append(1, ':'); // 必须紧跟参数使用1个冒号，如 "t:"
            } else {
                // do nothing
            }
        }
    }
    return shortOpt;
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
        default:
            ;
    }
}

UserCommand ClientParser::Parse(int32_t argc, char **argv)
{
    UserCommand userCommand;
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
        // 需打印help或version时，不进行其他操作
        if (userCommand.printHelpInfo || userCommand.printVersionInfo) {
            std::cout << "-h --help" << std::endl;
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