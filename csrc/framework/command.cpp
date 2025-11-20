// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "command.h"

#include <map>
#include <memory>

#include "process.h"
#include "utils.h"
#include "bit_field.h"
#include "analysis/memory_compare.h"
#include "analysis/dump.h"
#include "analysis/decompose_analyzer.h"
#include "analysis/inefficient_analyzer.h"

namespace MemScope {

void Command::Exec() const
{
    LOG_INFO("Msmemscope starts executing commands");
    
    if (userCommand_.config.enableCompare) {
        MemoryCompare::GetInstance(userCommand_.config).RunComparison(userCommand_.inputPaths);
        return;
    }

    Process::GetInstance(userCommand_.config).Launch(userCommand_.cmd);

    return;
}

}