// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "analyzer.h"

namespace Leaks {

Analyzer::Analyzer(const AnalysisConfig &config)
{
    config_ = config;
}

void Analyzer::Do(const EventRecord &record)
{
    return;
}

}