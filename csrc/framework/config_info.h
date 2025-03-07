// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef CONFIG_INFO_H
#define CONFIG_INFO_H

#include <cstdint>
#include <vector>
#include <string>

namespace Leaks {

constexpr uint8_t SELECTED_STEP_MAX_NUM = 5; // 先设定最多指定5个step的信息采集
constexpr const char *LEAKS_HEADERS = "Record Index,Timestamp(us),Event,Event Type,Process Id,Thread Id,Device Id,"
        "Kernel Index,Flag,Addr,Size(byte),Total Allocated(byte),Total Reserved(byte)\n";
constexpr const char *STEP_INTER_HEADERS = ",,Base,Compare\nName,Device Id,Allocated Memory(byte),"
        "Allocated Memory(byte),Diff Memory(byte)\n";

constexpr const char *OUTPUT_PATH = "leaksDumpResults";
constexpr const char *TRACE_FILE = "trace";
constexpr const char *DUMP_FILE = "dump";
constexpr const char *COMPARE_FILE = "compare";

enum class LevelType : uint8_t {
    LEVEL_0 = 0,
    LEVEL_1,
};

struct SelectedStepList {
    uint32_t stepIdList[SELECTED_STEP_MAX_NUM];
    uint8_t stepCount;
};

// 内存分析算法配置
struct AnalysisConfig {
    SelectedStepList stepList;
    bool enableCompare;
    bool inputCorrectPaths;
    bool outputCorrectPaths;
    LevelType levelType;
};

// 用于承载用户命令行参数的解析结果
struct UserCommand {
    bool printHelpInfo { false };
    bool printVersionInfo { false };
    AnalysisConfig config;
    std::vector<std::string> cmd;
    std::vector<std::string> inputPaths;
    std::string outputPath;
};

}
#endif