// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "client_parser.h"

#include <unistd.h>
#include <iostream>
#include <sys/stat.h>
#include <getopt.h>
#include <regex>
#include <algorithm>
#include "command.h"
#include "file.h"
#include "path.h"
#include "log.h"
#include "ustring.h"

namespace Leaks {

enum class OptVal : int32_t {
    SELECT_STEPS = 0,
    COMPARE,
    INPUT,
    OUTPUT,
    DATA_PARSING_LEVEL,
    LOG_LEVEL,
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
        "    --level=<level>                Set the data parsing level" << std::endl <<
        "                                   Level [1] for more detail info(default:0)" << std::endl <<
        "    --steps=1,2,3,...              Select the steps to collect memory information." << std::endl <<
        "                                   The input step numbers need to be separated by, or ，." << std::endl <<
        "                                   The maximum number of steps is 5" << std::endl <<
        "    --compare                      Enable memory data comparison." << std::endl <<
        "    --input=path1,path2            Paths to compare files, valid with compare command on" << std::endl <<
        "                                   The input paths need to be separated by, or ，." << std::endl <<
        "    --output=path                  The path to store the generated files." << std::endl <<
        "    --log-level                    Set log level to <level> [warn]." << std::endl;
}

void ShowVersion()
{
    ShowDescription();
    std::cout <<
        std::endl << "msleaks version " << "1.0" << "-" << __MSLEAKS_COMMIT_ID__ << std::endl;
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

    Utility::SetDirPath(userCommand.outputPath, std::string(OUTPUT_PATH));

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
        {"steps", required_argument, nullptr, static_cast<int32_t>(OptVal::SELECT_STEPS)},
        {"compare", no_argument, nullptr, static_cast<int32_t>(OptVal::COMPARE)},
        {"input", required_argument, nullptr, static_cast<int32_t>(OptVal::INPUT)},
        {"output", required_argument, nullptr, static_cast<int32_t>(OptVal::OUTPUT)},
        {"level", required_argument, nullptr, static_cast<int32_t>(OptVal::DATA_PARSING_LEVEL)},
        {"log-level", required_argument, nullptr, static_cast<int32_t>(OptVal::LOG_LEVEL)},
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

    auto parseFailed = [&userCommand](void) {
        std::cout << "[msleaks] ERROR: invalid steps input." << std::endl;
        userCommand.printHelpInfo = true;
    };

    while (it != end) {
        SelectedStepList &stepListInfo = userCommand.config.stepList;

        if (stepListInfo.stepCount >= SELECTED_STEP_MAX_NUM) {
            return parseFailed();
        }
        std::string step = it->str();
        if (!step.empty()) {
            if (!std::regex_match(step, numberPattern)) {
                return parseFailed();
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
            userCommand.inputPaths.emplace_back(path);
        }
        it++;
    }

    if (userCommand.inputPaths.size() != PATHSIZE) {
        std::cout << "[msleaks] ERROR: invalid paths input." << std::endl;
        userCommand.printHelpInfo = true;
    } else {
        userCommand.config.inputCorrectPaths = true;
    }
}

static void ParseOutputPath(const std::string param, UserCommand &userCommand)
{
    if (Utility::Strip(param).length() == 0) {
        std::cout << "[msleaks] WARN: empty path, use the default path leaksDumpResults." << std::endl;
        return;
    }

    Utility::Path path = Utility::Path{param};
    Utility::Path realPath = path.Resolved();

    if (!path.IsValidLength()) {
        std::cout << "[msleaks] WARN: invalid output path length, use the default path leaksDumpResults." << std::endl;
        return;
    }

    std::string pathStr = realPath.ToString();

    struct stat statbuf;
    if (lstat(pathStr.c_str(), &statbuf) == 0 && S_ISLNK(statbuf.st_mode)) {
        std::cout << "[msleaks] WARN: the path is link, use the default path leaksDumpResults." << std::endl;
        return;
    }

    std::regex pattern("(\\.|/|_|-|\\s|[~0-9a-zA-Z]|[\u4e00-\u9fa5])+");
    if (!std::regex_match(pathStr, pattern)) {
        std::cout << "[msleaks] WARN: invalid output path, use the default path leaksDumpResults." << std::endl;
        return;
    }

    userCommand.outputPath = pathStr;
}

static void ParseDataLevel(const std::string param, UserCommand &userCommand)
{
    std::vector<std::string> validLevelValues = {"0", "1"};

    uint8_t mode;
    if (std::find(validLevelValues.begin(), validLevelValues.end(), param) == validLevelValues.end()) {
        std::cout << "[msleaks] ERROR: --level param is invalid. "
                  << "optional value: 0,1, default:0" << std::endl;
        userCommand.printHelpInfo = true;
        return ;
    } else {
        mode = std::stoi(param);
        userCommand.config.levelType = static_cast<LevelType>(mode);
    }
}

static void ParseLogLv(const std::string &param, UserCommand &userCommand)
{
    const std::map<std::string, Utility::LogLv> logLevelMap = {
        {"info", Utility::LogLv::INFO},
        {"warn", Utility::LogLv::WARN},
        {"error", Utility::LogLv::ERROR},
    };
    auto it = logLevelMap.find(param);
    if (it == logLevelMap.end()) {
        std::cout << "[msleaks] ERROR: --log-level param is invalid"
                  << "LOG_LEVEL can only be set info,warn,error." << std::endl;
        userCommand.printHelpInfo = true;
    } else {
        auto logLevel = it->second;
        Utility::SetLogLevel(logLevel);
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
        case static_cast<int32_t>(OptVal::SELECT_STEPS):
            ParseSelectSteps(param, userCommand);
            break;
        case static_cast<int32_t>(OptVal::COMPARE):
            userCommand.config.enableCompare = true;
            break;
        case static_cast<int32_t>(OptVal::INPUT):
            ParseInputPaths(param, userCommand);
            break;
        case static_cast<int32_t>(OptVal::OUTPUT):
            ParseOutputPath(param, userCommand);
            break;
        case static_cast<int32_t>(OptVal::DATA_PARSING_LEVEL):
            ParseDataLevel(param, userCommand);
            break;
        case static_cast<int32_t>(OptVal::LOG_LEVEL):
            ParseLogLv(param, userCommand);
            break;
        default:
            ;
    }
}

void ClientParser::InitialUserCommand(UserCommand &userCommand)
{
    userCommand.config.stepList.stepCount = 0;
    userCommand.config.enableCompare = false;
    userCommand.config.inputCorrectPaths = false;
    userCommand.config.levelType = LevelType::LEVEL_0;
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