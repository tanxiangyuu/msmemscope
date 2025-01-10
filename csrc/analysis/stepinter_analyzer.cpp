// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "stepinter_analyzer.h"
#include <fstream>
#include <sstream>
#include "file.h"
#include "utils.h"

namespace Leaks {

std::vector<std::string> StepInterAnalyzer::SplitLineData(std::string line)
{
    std::vector<std::string> lineData;
    Utility::Split(line, std::back_inserter(lineData), ",");
    return lineData;
}

void StepInterAnalyzer::ReadCsvFile(const std::string &path, std::unordered_map<DEVICEID, CSV_FIELD_DATA> &data)
{
    if (!Utility::Exist(path)) {
        Utility::LogError("The leaks csv file path: %s does not exist!", path);
    }
    
    std::ifstream csvFile(path, std::ios::in);
    if (!csvFile.is_open()) {
        Utility::LogError("The path: %s open failed!", path);
    }

    std::string line;
    std::istringstream sin;

    // 读取表头
    getline(csvFile, line);
    sin.str(line);
    std::vector<std::string> headerData;
    headerData = SplitLineData(line);

    while (getline(csvFile, line)) {
        sin.str(line);
        std::vector<std::string> lineData;
        lineData = SplitLineData(line);
        std::unordered_map<std::string, std::string> tempLine;
        for (size_t index = 0; index < headerData.size(); ++index) {
            tempLine.insert({headerData[index], lineData[index]});
        }

        uint64_t deviceId = atoi(tempLine["deviceID"].c_str());
        data[deviceId].emplace_back(tempLine);
    }

    for (const auto& pair : data) {
        uint64_t deviceId = pair.first;
        // kernelName解析使用多线程，需要根据timeStamp排序保证顺序
        sort(data[deviceId].begin(), data[deviceId].end(), [](const std::unordered_map<std::string, std::string> &a,
             const std::unordered_map<std::string, std::string> &b) {
                return atoi(a.at("timeStamp").c_str()) < atoi(b.at("timeStamp").c_str());
        });
    }
    csvFile.close();
}

KERNELNAME_INDEX StepInterAnalyzer::ReadKernelLaunchData(const CSV_FIELD_DATA &data)
{
    KERNELNAME_INDEX result;
    for (size_t index = 0; index < data.size(); ++index) {
        auto lineData = data[index];
        if (lineData["type"] == "kernelLaunch") {
            result.emplace_back(std::make_pair(lineData["name"], index));
        }
    }
    return result;
}

void StepInterAnalyzer::GetKernelMemoryDiff(size_t index, const CSV_FIELD_DATA &data, TorchNouMemoryDiff &memDiff)
{
    std::unordered_map<std::string, std::string> preMemory;
    std::unordered_map<std::string, std::string> nextMemory;
    for (size_t i = index; i < data.size(); ++i) {
        auto lineData = data[i];
        if (lineData["type"] == "torch_npu") {
            nextMemory = lineData;
            break;
        }
    }
    for (int i = index; i >= 0; --i) {
        auto lineData = data[i];
        if (lineData["type"] == "torch_npu") {
            preMemory = lineData;
            break;
        }
    }
    if (preMemory.empty()) {
        preMemory["total_allocated"] = "0";
        preMemory["total_reserved"] = "0";
        preMemory["total_active"] = "0";
    }
    if (nextMemory.empty()) {
        nextMemory["total_allocated"] = "0";
        nextMemory["total_reserved"] = "0";
        nextMemory["total_active"] = "0";
    }
    int64_t pre;
    int64_t next;
    if (Utility::StrToInt64(pre, preMemory["total_allocated"]) &&
        Utility::StrToInt64(next, nextMemory["total_allocated"])) {
        memDiff.totalAllocated = Utility::GetSubResult(next, pre);
    } else {
        Utility::LogError("Totalallocated to uint64_t failed!");
    }
    if (Utility::StrToInt64(pre, preMemory["total_reserved"]) &&
        Utility::StrToInt64(next, nextMemory["total_reserved"])) {
        memDiff.totalReserved = Utility::GetSubResult(next, pre);
    } else {
        Utility::LogError("Totalreserved to uint64_t failed!");
    }
    if (Utility::StrToInt64(pre, preMemory["total_active"]) && Utility::StrToInt64(next, nextMemory["total_active"])) {
        memDiff.totalActive = Utility::GetSubResult(next, pre);
    } else {
        Utility::LogError("Totalactive to uint64_t failed!");
    }
}

bool StepInterAnalyzer::WriteCompareDataToCsv()
{
    std::string dirPath = "leaksDumpResults";
    std::string fileName = "stepintercompare" + Utility::GetDateStr() + ".csv";
    if (!Utility::CreateCsvFile(&compareFile, dirPath, fileName, headers)) {
        Utility::LogError("Create stepintercompare csv file failed!");
        return false;
    }

    for (const auto& pair : compareOut) {
        uint64_t deviceId = pair.first;
        std::reverse(compareOut[deviceId].begin(), compareOut[deviceId].end());

        for (const auto& str : compareOut[deviceId]) {
            fprintf(compareFile, "%s\n", str.c_str());
        }
    }

    return true;
}

void StepInterAnalyzer::SaveCompareKernelMemory(const DEVICEID deviceId,
    const std::pair<std::string, size_t> &kernelBase, const std::pair<std::string, size_t> &kernelCompare)
{
    std::string temp;
    std::string name;
    TorchNouMemoryDiff baseDiff;
    TorchNouMemoryDiff compareDiff;

    std::string baseMemDiff;
    if (!kernelBase.first.empty()) {
        name = kernelBase.first;
        GetKernelMemoryDiff(kernelBase.second, output[deviceId], baseDiff);
        baseMemDiff = std::to_string(baseDiff.totalAllocated) + "," + std::to_string(baseDiff.totalReserved)
        + "," + std::to_string(baseDiff.totalActive);
    } else {
        baseMemDiff = "null,null,null";
    }

    std::string compareMemDiff;
    if (!kernelCompare.first.empty()) {
        name = kernelCompare.first;
        GetKernelMemoryDiff(kernelCompare.second, outputCompare[deviceId], compareDiff);
        compareMemDiff = std::to_string(compareDiff.totalAllocated) + "," + std::to_string(compareDiff.totalReserved)
            + "," + std::to_string(compareDiff.totalActive);
    } else {
        compareMemDiff = "null,null,null";
    }

    temp += name;
    temp = temp + "," + std::to_string(deviceId) + "," + baseMemDiff + "," + compareMemDiff;

    int64_t diffTotalAllocated = compareDiff.totalAllocated - baseDiff.totalAllocated;
    int64_t diffTotalReserved = compareDiff.totalReserved - baseDiff.totalReserved;
    int64_t diffTotalActive = compareDiff.totalActive - baseDiff.totalActive;

    temp = temp + "," + std::to_string(diffTotalAllocated) + "," + std::to_string(diffTotalReserved)
            + "," + std::to_string(diffTotalActive);
    compareOut[deviceId].emplace_back(temp);
}

std::shared_ptr<PathNode> StepInterAnalyzer::buildPath(const KERNELNAME_INDEX &kernelIndexMap,
    const KERNELNAME_INDEX &kernelIndexCompareMap)
{
    const uint64_t n = kernelIndexMap.size();
    const uint64_t m = kernelIndexCompareMap.size();
    const uint64_t max = m + n + 1;
    const uint64_t size = 1 + 2 * max;
    const uint64_t middle = size / 2;
    std::vector<std::shared_ptr<PathNode>> diagonal(size, nullptr); // 存储每一步的最优路径位置
    diagonal[middle + 1] = std::make_shared<Snake>(0, -1);
    auto start_time = Utility::GetTimeMicroseconds();
    for (int64_t d = 0; d < max; ++d) {
        for (int64_t k = -d; k <= d; k += KSTEPSIZE) {
            auto end_time = Utility::GetTimeMicroseconds();
            if ((end_time - start_time) >= MAXLOOPTIME) {
                Utility::LogError("Analysis failed!Reaching maximum loop time limit!");
                break;
            }
            uint64_t kmiddle = middle + k;
            uint64_t kplus = kmiddle + 1;
            uint64_t kminus = kmiddle - 1;
            uint64_t i;
            std::shared_ptr<PathNode> prev;
            if ((k == -d) || (k != d && diagonal[kminus]->i < diagonal[kplus]->i)) { // 最优路径为从上往下走
                i = diagonal[kplus]->i;
                prev = diagonal[kplus];
            } else { // 最优路径为从左往右走
                i = diagonal[kminus]->i + 1;
                prev = diagonal[kminus];
            }
            uint64_t j = i - k;
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
}

void StepInterAnalyzer::buildDiff(std::shared_ptr<PathNode> path, const DEVICEID deviceId,
    const KERNELNAME_INDEX &kernelIndexMap, const KERNELNAME_INDEX &kernelIndexCompareMap)
{
    while (path && path->prev && path->prev->j >= 0) {
        if (path->IsSnake()) { // base kernelName = compare kernelName
            int endi = path->i;
            int begini = path->prev->i;

            int endj = path->j;
            int beginj = path->prev->j;
            for (int i = endi - 1, j = endj - 1; i >= begini, j >= beginj; --i, --j) {
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
        Utility::LogInfo("Empty kernelLaunch data!");
        return ;
    } else {
        auto pathNode = buildPath(kernelIndexMap, kernelIndexCompareMap);
        buildDiff(pathNode, deviceId, kernelIndexMap, kernelIndexCompareMap);
    }
}

void StepInterAnalyzer::StepInterOfflineCompare(const std::vector<std::string> &paths)
{
    std::string path = paths[0];
    std::string pathCompare = paths[1];
    
    ReadCsvFile(path, output);
    ReadCsvFile(pathCompare, outputCompare);

    for (const auto& pair : output) {
        uint64_t deviceId = pair.first;
        KERNELNAME_INDEX kernelIndexMap = ReadKernelLaunchData(output[deviceId]);
        KERNELNAME_INDEX kernelIndexCompareMap = ReadKernelLaunchData(outputCompare[deviceId]);
        MyersDiff(deviceId, kernelIndexMap, kernelIndexCompareMap);
    }

    if (WriteCompareDataToCsv()) {
        Utility::LogInfo("Stepinter memory data analysis end!");
    }
}

}
