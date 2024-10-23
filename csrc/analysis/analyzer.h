// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef ANALYZER_H
#define ANALYZER_H

#include "framework/config_info.h"
#include "framework/record_info.h"

namespace Leaks {
// Analyzer类主要用于将单条解析信息分发给合适的分析工具
class Analyzer {
public:
    explicit Analyzer(const AnalysisConfig &config);
    void Do(const EventRecord &record);
private:
    AnalysisConfig config_;
};
}

#endif
