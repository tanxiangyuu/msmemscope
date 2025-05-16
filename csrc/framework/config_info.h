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
constexpr const char *LEAKS_HEADERS = "ID,Event,Event Type,Name,Timestamp(us),Process Id,Thread Id,Device Id,"
        "Ptr,Attr";
constexpr const char *STEP_INTER_HEADERS = ",,Base,Compare\nName,Device Id,Allocated Memory(byte),"
        "Allocated Memory(byte),Diff Memory(byte)\n";
constexpr const char *TRACE_HEADERS = "FuncInfo,StartTime,EndTime,Thread Id,Process Id\n";
constexpr const char *OUTPUT_PATH = "leaksDumpResults";
constexpr const char *TRACE_FILE = "trace";
constexpr const char *DUMP_FILE = "dump";
constexpr const char *COMPARE_FILE = "compare";
constexpr uint16_t WATCH_OP_DIR_MAX_LENGTH = 255;

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

// dump数据结构
struct DumpContainer {
    uint64_t id;
    std::string event;
    std::string eventType;
    std::string name;
    uint64_t timeStamp;
    uint64_t pid;
    uint64_t tid;
    std::string deviceId;
    std::string addr;
    std::string callStack = "";
    std::string attr = "";
};

struct WatchConfig {
    bool isWatched;
    bool fullContent;
    char start[WATCH_OP_DIR_MAX_LENGTH];
    char end[WATCH_OP_DIR_MAX_LENGTH];
    uint32_t outputId;
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
    WatchConfig watchConfig;
    uint8_t levelType;
    uint8_t eventType;
    char outputDir[128];
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