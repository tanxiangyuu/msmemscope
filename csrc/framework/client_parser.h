// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef CLINET_PARSER_H
#define CLINET_PARSER_H

#include <cstdint>
#include "config_info.h"

namespace Leaks {
constexpr uint32_t PATHSIZE = 2;
constexpr uint32_t DEPTHCOUNT = 2;

// 用于解析用户命令行
class ClientParser {
public:
    ClientParser() = default;
    void Interpretor(int32_t argc, char **argv);
private:
    UserCommand Parse(int32_t argc, char **argv);
    void InitialUserCommand(UserCommand &userCommand);
};
}

#endif
