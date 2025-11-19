// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef CLINET_PARSER_H
#define CLINET_PARSER_H

#include <cstdint>
#include "config_info.h"

namespace MemScope {
constexpr uint32_t PATHSIZE = 2;
constexpr uint32_t DEPTHCOUNT = 2;

// 用于解析用户命令行
class ClientParser {
public:
    ClientParser() = default;
    void Interpretor(int32_t argc, char **argv);
    void InitialConfig(Config &config);
private:
    UserCommand Parse(int32_t argc, char **argv);
};

void ParseCallstack(const std::string &param, Config &config, bool &printHelpInfo);
void ParseDataLevel(const std::string param, Config &config, bool &printHelpInfo);
void ParseEventTraceType(const std::string param, Config &config, bool &printHelpInfo);
void ParseDevice(const std::string &param, Config &config, bool &printHelpInfo);
void ParseAnalysis(const std::string &param, Config &config, bool &printHelpInfo);
void ParseWatchConfig(const std::string param, Config &config, bool &printHelpInfo);
void ParseSelectSteps(const std::string &param, Config &config, bool &printHelpInfo);
void ParseDataFormat(const std::string &param, Config &config, bool &printHelpInfo);
void ParseCollectMode(const std::string &param, Config &config, bool &printHelpInfo);
void ParseOutputPath(const std::string param, Config &config, bool &printHelpInfo);
void SetEventDefaultConfig(Config &config);
void SetAnalysisDefaultConfig(Config &config);
void SetEffectiveConfig(Config &config);
}
#endif
