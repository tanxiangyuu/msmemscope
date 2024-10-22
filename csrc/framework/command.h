// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef FRAMEWORK_COMMAND_H
#define FRAMEWORK_COMMAND_H

#include <vector>
#include <string>
#include "config_info.h"

namespace Leaks {

// Command类主要针对解析后的命令进行处理，是串接流程的主要类
class Command {
public:
    explicit Command(const AnalysisConfig &config) : config_{config} {}
    void Exec(const std::vector<std::string> &execParams) const;
private:
    AnalysisConfig config_;
};

}

#endif
