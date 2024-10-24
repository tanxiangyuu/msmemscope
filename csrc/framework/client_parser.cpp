// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "client_parser.h"
#include "command.h"

namespace Leaks {

void ClientParser::Interpretor(int32_t argc, char **argv)
{
    auto userCommand = Parse(argc, argv);
    Command command {userCommand.config};
    command.Exec(userCommand.cmd);
    return;
}

UserCommand ClientParser::Parse(int32_t argc, char **argv)
{
    return UserCommand {};
}

}