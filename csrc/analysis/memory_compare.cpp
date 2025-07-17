// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "memory_compare.h"
#include <fstream>
#include <sstream>
#include "file.h"
#include "utils.h"
#include "config_info.h"
#include "record_info.h"
#include "ustring.h"

namespace Leaks {

MemoryCompare& MemoryCompare::GetInstance(Config config)
{
    static MemoryCompare instance(config);
    return instance;
}

MemoryCompare::MemoryCompare(Config config)
{
    config_ = config;
    std::string cStack = config.enableCStack ? ",Call Stack(C)" : "";
    std::string pyStack = config.enablePyStack ? ",Call Stack(Python)" : "";
    csvHeader_ = LEAKS_HEADERS + pyStack + cStack + "\n";
    SetDirPath();
}

// 用此方式依次读取CSV的每一行，不会被单格数据的逗号干扰
std::string MemoryCompare::ReadQuotedField(std::stringstream& ss)
{
    std::string field;
    if (ss.peek() == '"') {  // 检查是否以引号开头，如果是就跳过（因为其中可能存在逗号）
        ss.get();
        std::getline(ss, field, '"');

        // 处理转义引号
        size_t pos = 0;
        while ((pos = field.find("\"\"", pos)) != std::string::npos) {
            field.replace(pos, 2, "\"");
            pos += 1;
        }
        // 跳过可能的分隔符（逗号）
        if (ss.peek() == ',') {
            ss.get();
        }
    } else {
        std::getline(ss, field, ',');  // 普通字段
    }
    return field;
}

bool Compare(const std::unordered_map<std::string, std::string> &a,
    const std::unordered_map<std::string, std::string> &b)
{
    uint64_t compareA;
    uint64_t compareB;

    if (!Utility::StrToUint64(compareA, a.at("Timestamp(ns)"))) {
        LOG_WARN("StrToUint64 failed, the str is %s.", a.at("Timestamp(ns)").c_str());
        compareA = UINT64_MAX;
    }
    if (!Utility::StrToUint64(compareB, b.at("Timestamp(ns)"))) {
        LOG_WARN("StrToUint64 failed, the str is %s.", b.at("Timestamp(ns)").c_str());
        compareB = UINT64_MAX;
    }

    return compareA < compareB;
}

void MemoryCompare::ReadFile(std::string &path, std::unordered_map<DEVICEID, ORIGINAL_FILE_DATA> &data)
{
    std::vector<std::string> fileName;
    Utility::Split(path, std::back_inserter(fileName), ".");
    if (fileName.size() > 0 && fileName.back() == "csv") {
        LOG_INFO("Read csv file: %s.", path.c_str());
        ReadCsvFile(path, data);
        for (const auto& pair : data) {
            uint64_t deviceId = pair.first;
            // 需要根据timestamp排序保证顺序
            sort(data[deviceId].begin(), data[deviceId].end(), Compare);
        }
    } else {
        LOG_ERROR("The file %s is an unsupported format.", path.c_str());
    }
}

bool MemoryCompare::CheckCsvHeader(std::string &path, std::ifstream& file, std::vector<std::string> &headerData)
{
    if (!file.is_open()) {
        LOG_ERROR("The path: %s open failed!", path.c_str());
        return false;
    }
    std::string line;
    getline(file, line);

    if (line + "\n" != std::string(csvHeader_)) {
        return false;
    }

    Utility::Split(line, std::back_inserter(headerData), ",");
    return true;
}

bool IsSupportedFramework(const std::string& name)
{
    static const std::unordered_set<std::string> supportedFrameworks = {"PTA", "MINDSPORE"};
    return supportedFrameworks.find(name) != supportedFrameworks.end();
}

void MemoryCompare::ReadCsvFile(std::string &path, std::unordered_map<DEVICEID, ORIGINAL_FILE_DATA> &data)
{
    std::ifstream csvFile(path, std::ios::in);
    std::vector<std::string> headerData;
    if (!CheckCsvHeader(path, csvFile, headerData)) {
        LOG_ERROR("The headers of %s file is illegal!", path.c_str());
        return ;
    }
    std::string line;
    uint64_t countLine = 1;
    while (getline(csvFile, line)) {
        ++countLine;
        std::vector<std::string> lineData;
        std::stringstream ss(line);
        while (ss.good()) {
            std::string singleValue = ReadQuotedField(ss);
            Utility::ToSafeString(singleValue);
            lineData.emplace_back(singleValue);
        }
        if (lineData.size() != headerData.size()) {
            LOG_ERROR("The file %s on line %d is invalid!", path.c_str(), countLine);
            data.clear();
            return ;
        }
        std::unordered_map<std::string, std::string> tempLine;
        for (size_t index = 0; index < headerData.size(); ++index) {
            tempLine.insert({headerData[index], lineData[index]});
        }
        if (IsSupportedFramework(tempLine["Event Type"])) {
            if (framework_.empty()) {
                framework_ = tempLine["Event Type"];
            }
            if (framework_ != tempLine["Event Type"]) {
                LOG_ERROR("The content of the file %s is invalid.", path.c_str());
                data.clear();
                return ;
            }
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
    csvFile.close();
}

void MemoryCompare::ReadNameIndexData(const ORIGINAL_FILE_DATA &originData, NAME_WITH_INDEX &dataList)
{
    LOG_DEBUG("Read kernelLaunch/op data.");
    for (size_t index = 0; index < originData.size(); ++index) {
        auto lineData = originData[index];
        if (lineData["Event Type"] == "KERNEL_LAUNCH") {
            if (Utility::CheckStrIsStartsWithInvalidChar(lineData["Name"].c_str())) {
                LOG_ERROR("Name %s is invalid!", lineData["Name"].c_str());
                dataList.clear();
                return ;
            }
            dataList.emplace_back(std::make_pair(lineData["Name"], index));
        }
    }
}

void MemoryCompare::GetMemoryUsage(size_t index, const ORIGINAL_FILE_DATA &data, int64_t &memDiff)
{
    LOG_DEBUG("Get memorypool usage.");
    std::unordered_map<std::string, std::string> frameworkMemory;
    for (size_t i = index; i < data.size(); ++i) {
        auto lineData = data[i];
        if (lineData["Event Type"] == framework_) {
            frameworkMemory = lineData;
            break;
        }
    }

    if (frameworkMemory.empty()) {
        memDiff = 0;
        return ;
    }

    std::string attrKey = "size";
    std::string attrValue = Utility::ExtractAttrValueByKey(frameworkMemory["Attr"], attrKey);
    if (attrValue.empty()) {
        LOG_WARN("Attr has no \"size\" value");
        return ;
    }
    if (!Utility::StrToInt64(memDiff, attrValue)) {
        LOG_WARN("Alloc Size to int64_t failed!");
    }
}

bool MemoryCompare::WriteCompareDataToCsv()
{
    LOG_DEBUG("Write compare result data to csv file.");
    if (result_.empty()) {
        LOG_WARN("Empty comparison result data!");
        return false;
    }

    if (!Utility::CreateCsvFile(&compareFile_, dirPath_, fileNamePrefix_, std::string(STEP_INTER_HEADERS))) {
        LOG_ERROR("Create comparison csv file failed!");
        return false;
    }

    for (const auto& pair : result_) {
        uint64_t deviceId = pair.first;
        std::reverse(result_[deviceId].begin(), result_[deviceId].end());

        for (const auto& str : result_[deviceId]) {
            int fpRes = fprintf(compareFile_, "%s\n", str.c_str());
            if (fpRes < 0) {
                std::cout << "[msleaks] Error: Fail to write data to csv file, errno:" << fpRes << std::endl;
                return false;
            }
        }
    }

    return true;
}

void MemoryCompare::CalcuMemoryDiff(const DEVICEID deviceId, const std::pair<std::string, size_t> &baseData,
    const std::pair<std::string, size_t> &compareData)
{
    std::string temp;
    std::string name;
    int64_t baseAllocSize = 0;
    int64_t compareAllocSize = 0;

    std::string baseMemDiff;
    if (!baseData.first.empty()) {
        name = baseData.first;
        GetMemoryUsage(baseData.second, baseFileOriginData_[deviceId], baseAllocSize);
        baseMemDiff = std::to_string(baseAllocSize);
    } else {
        baseMemDiff = "N/A";
    }

    std::string compareMemDiff;
    if (!compareData.first.empty()) {
        name = compareData.first;
        GetMemoryUsage(compareData.second, compareFileOriginData_[deviceId], compareAllocSize);
        compareMemDiff = std::to_string(compareAllocSize);
    } else {
        compareMemDiff = "N/A";
    }

    temp += name;
    temp = temp + "," + std::to_string(deviceId) + "," + baseMemDiff + "," + compareMemDiff;

    int64_t diffAllocSize = Utility::GetSubResult(compareAllocSize, baseAllocSize);
    temp = temp + "," + std::to_string(diffAllocSize);
    result_[deviceId].emplace_back(temp);
}

std::shared_ptr<PathNode> MemoryCompare::BuildPath(const NAME_WITH_INDEX &baseLists,
    const NAME_WITH_INDEX &compareLists)
{
    LOG_DEBUG("Start to build myers path.");
    const int64_t n = static_cast<int64_t>(baseLists.size());
    const int64_t m = static_cast<int64_t>(compareLists.size());
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
                LOG_ERROR("Memory comparison build path failed! Reaching maximum loop time limit!");
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
            // 判断两个name是否相同
            while (i < n && j < m && (baseLists[i].first == compareLists[j].first)) {
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

void MemoryCompare::BuildDiff(std::shared_ptr<PathNode> path, const DEVICEID deviceId,
    const NAME_WITH_INDEX &baseLists, const NAME_WITH_INDEX &compareLists)
{
    LOG_DEBUG("Start to build myers diff.");
    if (path == nullptr) {
        LOG_WARN("Empty myers path!");
        return ;
    }
    auto start_time = Utility::GetTimeMicroseconds();
    while (path && path->prev && path->prev->j >= 0) {
        auto end_time = Utility::GetTimeMicroseconds();
        if ((end_time - start_time) >= MAXLOOPTIME) {
                LOG_ERROR("Memory compare build diff failed! Reaching maximum loop time limit!");
                break;
            }
        if (path->IsSnake()) { // base name = compare name
            int endi = path->i;

            int endj = path->j;
            int beginj = path->prev->j;
            for (int i = endi - 1, j = endj - 1; j >= beginj; --i, --j) {
                CalcuMemoryDiff(deviceId, baseLists[i], compareLists[j]);
            }
        } else {
            int i = path->i;
            int j = path->j;
            int prei = path->prev->i;
            if (prei < i) { // base name diff
                CalcuMemoryDiff(deviceId, baseLists[i - 1], {});
            } else { // compare name diff
                CalcuMemoryDiff(deviceId, {}, compareLists[j - 1]);
            }
        }
        path = path->prev;
    }
}

void MemoryCompare::MyersDiff(const DEVICEID deviceId, const NAME_WITH_INDEX &baseLists,
    const NAME_WITH_INDEX &compareLists)
{
    LOG_DEBUG("Start to compare with Myers algorithm.");
    if (baseLists.empty() && compareLists.empty()) {
        LOG_WARN("Empty kernelLaunch/op data!");
        return ;
    } else {
        auto pathNode = BuildPath(baseLists, compareLists);
        BuildDiff(pathNode, deviceId, baseLists, compareLists);
    }
}

void MemoryCompare::RunComparison(const std::vector<std::string> &paths)
{
    LOG_INFO("Start to compare memory data.");
    auto start_time = Utility::GetTimeMicroseconds();
    // 已在命令行输入处校验path长度
    std::string pathBase = paths[0];
    std::string pathCompare = paths[1];
    
    ReadFile(pathBase, baseFileOriginData_);
    ReadFile(pathCompare, compareFileOriginData_);

    if (baseFileOriginData_.empty() || compareFileOriginData_.empty()) {
        std::cout << "[msleaks] ERROR: Memory comparison failed!" << std::endl;
        return ;
    }

    for (const auto& pair : baseFileOriginData_) {
        uint64_t deviceId = pair.first;
        NAME_WITH_INDEX baseLists {};
        NAME_WITH_INDEX compareLists {};
        ReadNameIndexData(baseFileOriginData_[deviceId], baseLists);
        ReadNameIndexData(compareFileOriginData_[deviceId], compareLists);
        MyersDiff(deviceId, baseLists, compareLists);
    }

    if (!WriteCompareDataToCsv()) {
        std::cout << "[msleaks] ERROR: Memory comparison failed!" << std::endl;
    } else {
        auto end_time = Utility::GetTimeMicroseconds();
        LOG_INFO("The memory comparison has been completed "
            "in a total time of %.6f(s)", (end_time-start_time) / MICROSEC);
    }
    return ;
}

void MemoryCompare::SetDirPath()
{
    std::lock_guard<std::mutex> lock(fileMutex_);
    dirPath_ = Utility::g_dirPath + "/" + std::string(COMPARE_FILE);
}
}
