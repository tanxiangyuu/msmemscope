/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

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
