// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 
#include <gtest/gtest.h>
#include "command.h"
 
using namespace Leaks;
 
TEST(Command, run_ls_command_expect_success)
{
    std::vector<std::string> paramList = {"/bin/ls"};
    AnalysisConfig config;
    Command command(config);
    command.Exec(paramList);
}