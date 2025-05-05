// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "client_parser.h"

#include <unistd.h>
#include <iostream>
#include <sys/stat.h>
#include <getopt.h>
#include <regex>
#include <algorithm>
#include <unordered_map>
#include "command.h"
#include "file.h"
#include "path.h"
#include "log.h"
#include "ustring.h"
#include "bit_field.h"

namespace Leaks {

enum class OptVal : int32_t {
    SELECT_STEPS = 0,
    CALL_STACK,
    COMPARE,
    INPUT,
    OUTPUT,
    DATA_TRACE_LEVEL,
    LOG_LEVEL,
    EVENT_TRACE_TYPE,
};
constexpr uint16_t INPUT_STR_MAX_LEN = 4096;

void ShowDescription()
{
    std::cout <<
        "msleaks(MindStudio Leaks) is part of MindStudio Memory analysis Tools." << std::endl;
}

void ShowHelpInfo()
{
    ShowDescription();
    std::cout << std::endl
        << "Usage: msleaks <option(s)> prog-and-args" << std::endl << std::endl
        << "  basic user options, with default in [ ]:" << std::endl
        << "    -h --help                                Show this message." << std::endl
        << "    -v --version                             Show version." << std::endl
        << "    --level=<level>                          Set the data trace level." << std::endl
        << "                                             Level [1] for kernel, Level [0] for op(default:0)."
        << std::endl
        << "                                             You can choose both separated by, or ，." << std::endl
        << "    --events=event1,event2                   Set the trace event type." << std::endl
        << "                                             You can combine any of the following options:" << std::endl
        << "                                             alloc,free,launch,access (default:alloc,free,launch)."
        << std::endl
        << "    --steps=1,2,3,...                        Select the steps to collect memory information." << std::endl
        << "                                             The input step numbers need to be separated by, or ，."
        << std::endl
        << "                                             The maximum number of steps is 5." << std::endl
        << "    --call-stack=<c/python>[:<Depth>],...    Enable C,Python call stack collection for memory event."
        << std::endl
        << "                                             Select the maximum depth of the collected call stack."
        << std::endl
        << "                                             If no depth is specified, use default depth(50)." << std::endl
        << "                                             e.g. --call-stack=c:20,python:10" << std::endl
        << "                                                  --call-stack=python" << std::endl
        << "                                             The input params need to be separated by, or ，." << std::endl
        << "    --compare                                Enable memory data comparison." << std::endl
        << "    --input=path1,path2                      Paths to compare files, valid with compare command on."
        << std::endl
        << "                                             The input paths need to be separated by, or ，." << std::endl
        << "    --output=path                            The path to store the generated files." << std::endl
        << "    --log-level                              Set log level to <level> [warn]." << std::endl;
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

    if (!userCommand.config.outputCorrectPaths) {
        std::cout << "Please use correct output path!" << std::endl;
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
        ShowHelpInfo();
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
        {"call-stack", required_argument, nullptr, static_cast<int32_t>(OptVal::CALL_STACK)},
        {"compare", no_argument, nullptr, static_cast<int32_t>(OptVal::COMPARE)},
        {"input", required_argument, nullptr, static_cast<int32_t>(OptVal::INPUT)},
        {"output", required_argument, nullptr, static_cast<int32_t>(OptVal::OUTPUT)},
        {"level", required_argument, nullptr, static_cast<int32_t>(OptVal::DATA_TRACE_LEVEL)},
        {"events", required_argument, nullptr, static_cast<int32_t>(OptVal::EVENT_TRACE_TYPE)},
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
            if (!Utility::StrToUint32(stepListInfo.stepIdList[stepListInfo.stepCount], it->str())) {
                return parseFailed();
            }
            stepListInfo.stepCount++;
        }
        
        it++;
    }

    return;
}

static bool CheckIsValidDepthInfo(const std::string &param, UserCommand &userCommand)
{
    size_t pos = param.find(':');
    std::string callType = param.substr(0, pos);
    std::regex numberPattern(R"(^(0|[1-9]\d*)$)");
    uint32_t depth = DEFAULT_CALL_STACK_DEPTH;

    if (pos != std::string::npos) {
        std::string depthStr = param.substr(pos + 1);
        if (depthStr.empty() || !std::regex_match(depthStr, numberPattern) || !Utility::StrToUint32(depth, depthStr)) {
            return false;
        }
    }
    if (callType == "python" && !userCommand.config.enablePyStack) {
        userCommand.config.enablePyStack = true;
        userCommand.config.pyStackDepth = depth;
    } else if (callType == "c" && !userCommand.config.enableCStack) {
        userCommand.config.enableCStack = true;
        userCommand.config.cStackDepth = depth;
    } else {
        return false;
    }
    return true;
}

static void ParseCallstack(const std::string &param, UserCommand &userCommand)
{
    std::regex dividePattern(R"([，,])");
    std::sregex_token_iterator  it(param.begin(), param.end(), dividePattern, -1);
    std::sregex_token_iterator  end;

    auto parseFailed = [&userCommand](void) {
        std::cout << "[msleaks] ERROR: invalid call-stack depth input." << std::endl;
        userCommand.printHelpInfo = true;
    };

    while (it != end) {
        std::string depthStr = it->str();
        if (!CheckIsValidDepthInfo(depthStr, userCommand)) {
            return parseFailed();
        }
        it++;
    }
    return;
}

static void ParseInputPaths(const std::string param, UserCommand &userCommand)
{
    if (param.length() > INPUT_STR_MAX_LEN) {
        std::cout << "[msleaks] Error: Parameter --input length exceeds the maximum length:"
                  << INPUT_STR_MAX_LEN << "." << std::endl;
        return;
    }
    std::regex pattern(R"([，,])");
    std::sregex_token_iterator  it(param.begin(), param.end(), pattern, -1);
    std::sregex_token_iterator  end;

    while (it != end) {
        std::string path = it->str();
        if (!path.empty() && Utility::CheckIsValidInputPath(path) && Utility::IsFileSizeSafe(path)) {
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
    if (param.length() > PATH_MAX) {
        std::cout << "[msleaks] Error: Parameter --output length exceeds the maximum length:"
                  << PATH_MAX << " output path will be set to default(./leaksDumpResults)." << std::endl;
        return;
    }
    if (Utility::Strip(param).length() == 0) {
        userCommand.config.outputCorrectPaths = false;
        std::cout << "[msleaks] WARN: empty output path." << std::endl;
        return;
    }

    Utility::Path path = Utility::Path{param};
    Utility::Path realPath = path.Resolved();
    std::string pathStr = realPath.ToString();

    std::regex pattern("(\\.|/|_|-|\\s|[~0-9a-zA-Z]|[\u4e00-\u9fa5])+");
    if (!Utility::CheckIsValidOutputPath(pathStr) || !std::regex_match(pathStr, pattern)) {
        userCommand.config.outputCorrectPaths = false;
        std::cout << "[msleaks] WARN: invalid output path." << std::endl;
        return;
    }

    userCommand.outputPath = pathStr;
}

static void ParseDataLevel(const std::string param, UserCommand &userCommand)
{
    std::regex dividePattern(R"([，,])");
    std::sregex_token_iterator it(param.begin(), param.end(), dividePattern, -1);
    std::sregex_token_iterator end;

    std::regex numberPattern(R"(^[01]$)");

    auto parseFailed = [&userCommand](void) {
        std::cout << "[msleaks] ERROR: invalid data trace level input." << std::endl;
        userCommand.printHelpInfo = true;
    };

    BitField<decltype(userCommand.config.levelType)> levelBit;

    while (it != end) {
        std::string level = it->str();
        if (!level.empty()) {
            if (!std::regex_match(level, numberPattern)) {
                return parseFailed();
            }
            if (level == "0") {
                levelBit.setBit(static_cast<size_t>(LevelType::LEVEL_OP));
            } else if (level == "1") {
                levelBit.setBit(static_cast<size_t>(LevelType::LEVEL_KERNEL));
            }
        }
        it++;
    }

    userCommand.config.levelType = levelBit.getValue();
    return;
}

static void ParseEventTraceType(const std::string param, UserCommand &userCommand)
{
    std::regex dividePattern(R"([，,])");
    std::sregex_token_iterator it(param.begin(), param.end(), dividePattern, -1);
    std::sregex_token_iterator end;

    auto parseFailed = [&userCommand](void) {
        std::cout << "[msleaks] ERROR: invalid event trace type input." << std::endl;
        userCommand.printHelpInfo = true;
    };

    BitField<decltype(userCommand.config.eventType)> eventsTypeBit;

    std::unordered_map<std::string, EventType> eventsMp = {
        {"alloc", EventType::ALLOC_EVENT},
        {"free", EventType::FREE_EVENT},
        {"launch", EventType::LAUNCH_EVENT},
        {"access", EventType::ACCESS_EVENT}
    };
    while (it != end) {
        std::string event = it->str();
        if (!event.empty()) {
            if (eventsMp.count(event)) {
                eventsTypeBit.setBit(static_cast<size_t>(eventsMp[event]));
            } else {
                return parseFailed();
            }
        }
        it++;
    }

    userCommand.config.eventType = eventsTypeBit.getValue();
    return;
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
        std::cout << "[msleaks] ERROR: --log-level param is invalid. "
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
        case static_cast<int32_t>(OptVal::CALL_STACK):
            ParseCallstack(param, userCommand);
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
        case static_cast<int32_t>(OptVal::DATA_TRACE_LEVEL):
            ParseDataLevel(param, userCommand);
            break;
        case static_cast<int32_t>(OptVal::EVENT_TRACE_TYPE):
            ParseEventTraceType(param, userCommand);
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
    userCommand.config.enableCStack = false;
    userCommand.config.enablePyStack = false;
    userCommand.config.inputCorrectPaths = false;
    userCommand.config.outputCorrectPaths = true;
    userCommand.config.cStackDepth = 0;
    userCommand.config.pyStackDepth = 0;
    userCommand.config.levelType = 0;

    BitField<decltype(userCommand.config.eventType)> eventBit;
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::LAUNCH_EVENT));
    userCommand.config.eventType = eventBit.getValue();
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