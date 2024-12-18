// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef CONFIG_INFO_H
#define CONFIG_INFO_H

#include <cstdint>
#include <vector>
#include <string>

namespace Leaks {

// 内存分析算法配置
struct AnalysisConfig {
    bool parseKernelName; // 解析kernelname的开关
};

// 用于承载用户命令行参数的解析结果
struct UserCommand {
    bool printHelpInfo { false };
    bool printVersionInfo { false };
    AnalysisConfig config;
    std::vector<std::string> cmd;
};

}
#endif