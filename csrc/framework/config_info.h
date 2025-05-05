// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef CONFIG_INFO_H
#define CONFIG_INFO_H

#include <cstdint>
#include <vector>
#include <string>

namespace Leaks {

constexpr uint8_t SELECTED_STEP_MAX_NUM = 5;  // 先设定最多指定5个step的信息采集
constexpr uint8_t DEFAULT_CALL_STACK_DEPTH = 50;
constexpr uint8_t SKIP_DEPTH = 2;
constexpr const char *LEAKS_HEADERS = "Record Index,Timestamp(us),Event,Event Type,Process Id,Thread Id,Device Id,"
        "Kernel Index,Flag,Addr,Size(byte),Total Allocated(byte),Total Reserved(byte)";
constexpr const char *STEP_INTER_HEADERS = ",,Base,Compare\nName,Device Id,Allocated Memory(byte),"
        "Allocated Memory(byte),Diff Memory(byte)\n";

constexpr const char *OUTPUT_PATH = "leaksDumpResults";
constexpr const char *TRACE_FILE = "trace";
constexpr const char *DUMP_FILE = "dump";
constexpr const char *COMPARE_FILE = "compare";

// level type可以多选，每一种type占一个bit位
enum class LevelType : uint8_t {
    LEVEL_OP = 0,
    LEVEL_KERNEL = 1,
};

// event type可以多选，每一种type占一个bit位
enum class EventType : uint8_t {
    ALLOC_EVENT = 0,
    FREE_EVENT = 1,
    LAUNCH_EVENT = 2,
    ACCESS_EVENT = 3,
};

struct SelectedStepList {
    uint32_t stepIdList[SELECTED_STEP_MAX_NUM];
    uint8_t stepCount;
};

// 内存分析算法配置
struct Config {
    SelectedStepList stepList;
    bool enableCompare;
    bool enableCStack;
    bool enablePyStack;
    uint32_t cStackDepth;
    uint32_t pyStackDepth;
    bool inputCorrectPaths;
    bool outputCorrectPaths;
    uint8_t levelType;
    uint8_t eventType;
};

// 用于承载用户命令行参数的解析结果
struct UserCommand {
    bool printHelpInfo { false };
    bool printVersionInfo { false };
    Config config;
    std::vector<std::string> cmd;
    std::vector<std::string> inputPaths;
    std::string outputPath;
};

}
#endif