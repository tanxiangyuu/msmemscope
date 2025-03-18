// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "stepinter_analyzer.h"
#include <fstream>
#include <sstream>
#include "file.h"
#include "utils.h"
#include "config_info.h"
#include "record_info.h"

namespace Leaks {

StepInterAnalyzer& StepInterAnalyzer::GetInstance()
{
    static StepInterAnalyzer instance;
    return instance;
}

StepInterAnalyzer::StepInterAnalyzer()
{
    SetDirPath();
}

std::vector<std::string> StepInterAnalyzer::SplitLineData(std::string line)
{
    std::vector<std::string> lineData;
    Utility::Split(line, std::back_inserter(lineData), ",");
    return lineData;
}

bool Compare(const std::unordered_map<std::string, std::string> &a,
    const std::unordered_map<std::string, std::string> &b)
{
    uint64_t compareA;
    uint64_t compareB;
    if (!Utility::StrToUint64(compareA, a.at("Timestamp(us)"))) {
        LOG_WARN("StrToUint64 failed, the str is %s.", a.at("Timestamp(us)").c_str());
        compareA = UINT64_MAX;
    }
    if (!Utility::StrToUint64(compareB, b.at("Timestamp(us)"))) {
        LOG_WARN("StrToUint64 failed, the str is %s.", b.at("Timestamp(us)").c_str());
        compareB = UINT64_MAX;
    }

    return compareA < compareB;
}

void StepInterAnalyzer::ReadCsvFile(std::string &path, std::unordered_map<DEVICEID, CSV_FIELD_DATA> &data)
{
    if (!Utility::CheckIsValidPath(path) || !Utility::IsFileExist(path)) {
        return ;
    }
    std::ifstream csvFile(path, std::ios::in);
    if (!csvFile.is_open()) {
        LOG_ERROR("The path: %s open failed!", path.c_str());
    }

    std::string line;
    std::istringstream sin;

    getline(csvFile, line);
    sin.str(line);
    if ((line + "\n") != std::string(LEAKS_HEADERS)) {
        LOG_ERROR("The headers of %s file is illegal!", path.c_str());
        return ;
    }
    std::vector<std::string> headerData;
    headerData = SplitLineData(line);
    uint64_t countLine = 1;
    while (getline(csvFile, line)) {
        sin.str(line);
        ++countLine;
        std::vector<std::string> lineData;
        lineData = SplitLineData(line);
        if (lineData.size() != headerData.size()) {
            LOG_ERROR("The file %s on line %d is invalid!", path.c_str(), countLine);
            data.clear();
            return ;
        }
        std::unordered_map<std::string, std::string> tempLine;
        for (size_t index = 0; index < headerData.size(); ++index) {
            tempLine.insert({headerData[index], lineData[index]});
        }
        uint64_t deviceId;
        if (tempLine["Device Id"] == std::to_string(GD_INVALID_NUM) || tempLine["Device Id"] == "host" ||
            tempLine["Device Id"] == "N/A") {
            continue;
        }
        if (!Utility::StrToUint64(deviceId, tempLine["Device Id"])) {
            LOG_WARN("StrToUint64 failed, the str is %s.", tempLine["Device Id"].c_str());
            continue;
        }
        data[deviceId].emplace_back(tempLine);
    }

    for (const auto& pair : data) {
        uint64_t deviceId = pair.first;
        // kernelName解析使用多线程，需要根据timeStamp排序保证顺序
        sort(data[deviceId].begin(), data[deviceId].end(), Compare);
    }
    csvFile.close();
}

KERNELNAME_INDEX StepInterAnalyzer::ReadKernelLaunchData(const CSV_FIELD_DATA &data)
{
    KERNELNAME_INDEX result;
    for (size_t index = 0; index < data.size(); ++index) {
        auto lineData = data[index];
        if (lineData["Event"] == "kernelLaunch") {
            result.emplace_back(std::make_pair(lineData["Event Type"], index));
        }
    }
    return result;
}

void StepInterAnalyzer::GetKernelMemoryDiff(size_t index, const CSV_FIELD_DATA &data, int64_t &memDiff)
{
    std::unordered_map<std::string, std::string> frameworkMemory;
    for (size_t i = index; i < data.size(); ++i) {
        auto lineData = data[i];
        if (lineData["Event"] == "pytorch") {
            frameworkMemory = lineData;
            break;
        }
    }

    if (frameworkMemory.empty()) {
        memDiff = 0;
        return ;
    }

    if (!Utility::StrToInt64(memDiff, frameworkMemory["Size(byte)"])) {
        LOG_WARN("Alloc Size to int64_t failed!");
    }
}

bool StepInterAnalyzer::WriteCompareDataToCsv()
{
    if (compareOut_.empty()) {
        LOG_WARN("Empty stepinter compare data!");
        return false;
    }

    if (!Utility::CreateCsvFile(&compareFile_, dirPath_, fileNamePrefix_, std::string(STEP_INTER_HEADERS))) {
        LOG_ERROR("Create stepintercompare csv file failed!");
        return false;
    }

    for (const auto& pair : compareOut_) {
        uint64_t deviceId = pair.first;
        std::reverse(compareOut_[deviceId].begin(), compareOut_[deviceId].end());

        for (const auto& str : compareOut_[deviceId]) {
            fprintf(compareFile_, "%s\n", str.c_str());
        }
    }

    return true;
}

void StepInterAnalyzer::SaveCompareKernelMemory(const DEVICEID deviceId,
    const std::pair<std::string, size_t> &kernelBase, const std::pair<std::string, size_t> &kernelCompare)
{
    std::string temp;
    std::string name;
    int64_t baseAllocSize = 0;
    int64_t compareAllocSize = 0;

    std::string baseMemDiff;
    if (!kernelBase.first.empty()) {
        name = kernelBase.first;
        GetKernelMemoryDiff(kernelBase.second, output_[deviceId], baseAllocSize);
        baseMemDiff = std::to_string(baseAllocSize);
    } else {
        baseMemDiff = "N/A";
    }

    std::string compareMemDiff;
    if (!kernelCompare.first.empty()) {
        name = kernelCompare.first;
        GetKernelMemoryDiff(kernelCompare.second, outputCompare_[deviceId], compareAllocSize);
        compareMemDiff = std::to_string(compareAllocSize);
    } else {
        compareMemDiff = "N/A";
    }

    temp += name;
    temp = temp + "," + std::to_string(deviceId) + "," + baseMemDiff + "," + compareMemDiff;

    int64_t diffAllocSize = Utility::GetSubResult(compareAllocSize, baseAllocSize);
    temp = temp + "," + std::to_string(diffAllocSize);
    compareOut_[deviceId].emplace_back(temp);
}

std::shared_ptr<PathNode> StepInterAnalyzer::buildPath(const KERNELNAME_INDEX &kernelIndexMap,
    const KERNELNAME_INDEX &kernelIndexCompareMap)
{
    const int64_t n = static_cast<int64_t>(kernelIndexMap.size());
    const int64_t m = static_cast<int64_t>(kernelIndexCompareMap.size());
    const int64_t max = m + n + 1;
    const int64_t size = 1 + 2 * max;
    const int64_t middle = size / 2;
    std::vector<std::shared_ptr<PathNode>> diagonal(size, nullptr); // 存储每一步的最优路径位置
    diagonal[middle + 1] = std::make_shared<Snake>(0, -1);
    auto start_time = Utility::GetTimeMicroseconds();
    for (int64_t d = 0; d < max; ++d) {
        for (int64_t k = -d; k <= d; k += KSTEPSIZE) {
            auto end_time = Utility::GetTimeMicroseconds();
            if ((end_time - start_time) >= MAXLOOPTIME) {
                LOG_ERROR("Stepinter analyze build path failed! Reaching maximum loop time limit!");
                break;
            }
            int64_t kmiddle = middle + k;
            int64_t kplus = kmiddle + 1;
            int64_t kminus = kmiddle - 1;
            int64_t i;
            std::shared_ptr<PathNode> prev;
            if ((k == -d) || (k != d && diagonal[kminus]->i < diagonal[kplus]->i)) { // 最优路径为从上往下走
                i = diagonal[kplus]->i;
                prev = diagonal[kplus];
            } else { // 最优路径为从左往右走
                i = diagonal[kminus]->i + 1;
                prev = diagonal[kminus];
            }
            int64_t j = i - k;
            diagonal[kminus] = nullptr;
            std::shared_ptr<PathNode> node = std::make_shared<DiffNode>(i, j, prev);
            // 判断两个kernelname是否相同
            while (i < n && j < m && (kernelIndexMap[i].first == kernelIndexCompareMap[j].first)) {
                ++i;
                ++j;
            }
            if (i > node->i) { // 对角线节点更新为snake
                node = std::make_shared<Snake>(i, j, node);
            }
            diagonal[kmiddle] = node;
            if (i >= n && j >= m) { // 达到终点，返回节点
                return diagonal[kmiddle];
            }
        }
    }
    return nullptr;
}

void StepInterAnalyzer::buildDiff(std::shared_ptr<PathNode> path, const DEVICEID deviceId,
    const KERNELNAME_INDEX &kernelIndexMap, const KERNELNAME_INDEX &kernelIndexCompareMap)
{
    if (path == nullptr) {
        LOG_WARN("Empty stepinter myers path!");
        return ;
    }
    auto start_time = Utility::GetTimeMicroseconds();
    while (path && path->prev && path->prev->j >= 0) {
        auto end_time = Utility::GetTimeMicroseconds();
        if ((end_time - start_time) >= MAXLOOPTIME) {
                LOG_ERROR("Stepinter analyze build diff failed! Reaching maximum loop time limit!");
                break;
            }
        if (path->IsSnake()) { // base kernelName = compare kernelName
            int endi = path->i;

            int endj = path->j;
            int beginj = path->prev->j;
            for (int i = endi - 1, j = endj - 1; j >= beginj; --i, --j) {
                SaveCompareKernelMemory(deviceId, kernelIndexMap[i], kernelIndexCompareMap[j]);
            }
        } else {
            int i = path->i;
            int j = path->j;
            int prei = path->prev->i;
            if (prei < i) { // base kernelName diff
                SaveCompareKernelMemory(deviceId, kernelIndexMap[i - 1], {});
            } else { // compare kernelName diff
                SaveCompareKernelMemory(deviceId, {}, kernelIndexCompareMap[j - 1]);
            }
        }
        path = path->prev;
    }
}

void StepInterAnalyzer::MyersDiff(const DEVICEID deviceId, const KERNELNAME_INDEX &kernelIndexMap,
    const KERNELNAME_INDEX &kernelIndexCompareMap)
{
    if (kernelIndexMap.empty() && kernelIndexCompareMap.empty()) {
        LOG_WARN("Empty kernelLaunch data!");
        return ;
    } else {
        auto pathNode = buildPath(kernelIndexMap, kernelIndexCompareMap);
        buildDiff(pathNode, deviceId, kernelIndexMap, kernelIndexCompareMap);
    }
}

void StepInterAnalyzer::StepInterCompare(const std::vector<std::string> &paths)
{
    LOG_INFO("Start to analyze stepinter memory data.");
    auto start_time = Utility::GetTimeMicroseconds();
    std::string path = paths[0];
    std::string pathCompare = paths[1];
    
    ReadCsvFile(path, output_);
    ReadCsvFile(pathCompare, outputCompare_);

    if (output_.empty() || outputCompare_.empty()) {
        std::cout << "[msleaks] ERROR: Stepinter analyze failed!" << std::endl;
        return ;
    }

    for (const auto& pair : output_) {
        uint64_t deviceId = pair.first;
        KERNELNAME_INDEX kernelIndexMap = ReadKernelLaunchData(output_[deviceId]);
        KERNELNAME_INDEX kernelIndexCompareMap = ReadKernelLaunchData(outputCompare_[deviceId]);
        MyersDiff(deviceId, kernelIndexMap, kernelIndexCompareMap);
    }

    if (!WriteCompareDataToCsv()) {
        std::cout << "[msleaks] ERROR: Stepinter analyze failed!" << std::endl;
    } else {
        auto end_time = Utility::GetTimeMicroseconds();
        LOG_INFO("The stepinter memory analysis has been completed"
            "in a total time of %.6f(s)", (end_time-start_time) / MICROSEC);
    }
    return ;
}

void StepInterAnalyzer::SetDirPath()
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    dirPath_ = Utility::g_dirPath + "/" + std::string(COMPARE_FILE);
}
}
