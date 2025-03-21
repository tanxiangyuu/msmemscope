// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef STEPINTER_ANALYZER_H
#define STEPINTER_ANALYZER_H

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>
#include <dirent.h>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>
#include <mutex>
#include "log.h"
#include "ustring.h"
#include "framework/config_info.h"

namespace Leaks {
constexpr uint32_t KSTEPSIZE = 2;
constexpr uint32_t MAXLOOPTIME = 60 * 1000000; // 最大处理时间1分钟
constexpr double MICROSEC = 1000000.0;

struct TorchNouMemoryDiff {
    int64_t totalAllocated = 0;
    int64_t totalReserved = 0;
    int64_t totalActive = 0;
};

using DEVICEID = uint64_t;
using CSV_FIELD_DATA = std::vector<std::unordered_map<std::string, std::string>>;
using KERNELNAME_INDEX = std::vector<std::pair<std::string, size_t>>;

/* PathNode基类
 * 1. 存储最优节点的坐标(i , j)，以及前驱节点prev
 * 2. 使用IsSnake函数标记是否为对角线节点
*/
class PathNode {
public:
    int i, j;
    std::shared_ptr<PathNode> prev;

    PathNode(int i, int j, std::shared_ptr<PathNode> prev = nullptr): i(i), j(j), prev(std::move(prev)) {}

    virtual ~PathNode() = default;

    virtual bool IsSnake() const = 0;
};

/* Snake类
 * 1. 存储最优节点的坐标(i , j)，以及前驱节点prev
 * 2. 该节点为对角线节点
*/
class Snake : public PathNode {
public:
    Snake(int i, int j, std::shared_ptr<PathNode> prev = nullptr)
        : PathNode(i, j, std::move(prev)) {}

    bool IsSnake() const override
    {
        return true;
    }
};

/* DiffNode类
 * 1. 存储最优节点的坐标(i , j)，以及前驱节点prev
 * 2. 该节点不是对角线节点
*/
class DiffNode : public PathNode {
public:
    DiffNode(int i, int j, std::shared_ptr<PathNode> prev = nullptr)
        : PathNode(i, j, std::move(prev)) {}

    bool IsSnake() const override
    {
        return false;
    }
};

class StepInterAnalyzer {
public:
    static StepInterAnalyzer& GetInstance();
    void StepInterCompare(const std::vector<std::string> &paths);
private:
    StepInterAnalyzer();
    void SetDirPath();
    std::vector<std::string> SplitLineData(std::string line);
    void ReadCsvFile(std::string &path, std::unordered_map<DEVICEID, CSV_FIELD_DATA> &data);
    bool ReadKernelLaunchData(const CSV_FIELD_DATA &data, KERNELNAME_INDEX &result);
    void GetKernelMemoryDiff(size_t index, const CSV_FIELD_DATA &data, int64_t &memDiff);
    void SaveCompareKernelMemory(const DEVICEID deviceId, const std::pair<std::string, size_t> &kernelBase,
        const std::pair<std::string, size_t> &kernelCompare);
    std::shared_ptr<PathNode> buildPath(const KERNELNAME_INDEX &kernelIndexMap,
        const KERNELNAME_INDEX &kernelIndexCompareMap);
    void buildDiff(std::shared_ptr<PathNode> path, const DEVICEID deviceId, const KERNELNAME_INDEX &kernelIndexMap,
        const KERNELNAME_INDEX &kernelIndexCompareMap);
    bool WriteCompareDataToCsv();
    void MyersDiff(const DEVICEID deviceId, const KERNELNAME_INDEX &kernelIndexMap,
        const KERNELNAME_INDEX &kernelIndexCompareMap);
    FILE* compareFile_ = nullptr;
    std::unordered_map<DEVICEID, CSV_FIELD_DATA> output_;
    std::unordered_map<DEVICEID, CSV_FIELD_DATA> outputCompare_;
    std::unordered_map<DEVICEID, std::vector<std::string>> compareOut_;
    std::string fileNamePrefix_ = "stepintercompare_";
    std::string dirPath_;
    std::mutex fileMutex_;
};

}

#endif