// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 
#include <gtest/gtest.h>
#include "command.h"
 
using namespace Leaks;
 
TEST(Command, run_ls_command_expect_success)
{
    UserCommand useCommand;
    useCommand.cmd = {{"/bin/ls"}};
    useCommand.config.enableCompare = false;
    useCommand.config.dataFormat = 0;
    Command command(useCommand);
    command.Exec();
}