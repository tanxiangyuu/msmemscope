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

namespace Leaks {

void Command::Exec() const
{
    LOG_INFO("Msleaks starts executing commands");

    if (userCommand_.config.enableCompare) {
        MemoryCompare::GetInstance(userCommand_.config).RunComparison(userCommand_.inputPaths);
        return;
    }

    auto config = userCommand_.config;
    BitField<decltype(config.analysisType)> analysisType(config.analysisType);
    if (analysisType.checkBit(static_cast<size_t>(AnalysisType::DECOMPOSE_ANALYSIS))) {
        DecomposeAnalyzer::GetInstance();
    }
    Dump::GetInstance(userCommand_.config);

    Process process(userCommand_.config);
    process.Launch(userCommand_.cmd);

    return;
}

}