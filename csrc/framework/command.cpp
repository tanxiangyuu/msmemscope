// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "command.h"
#include <map>
#include "process.h"
#include "utils.h"
#include "analysis/stepinter_analyzer.h"

namespace Leaks {

void Command::Exec() const
{
    LOG_INFO("Msleaks starts executing commands");

    if (userCommand_.config.enableCompare) {
        StepInterAnalyzer::GetInstance(userCommand_.config).StepInterCompare(userCommand_.inputPaths);
        return;
    }
    
    Process process(userCommand_.config);
    process.Launch(userCommand_.cmd);

    return;
}

}