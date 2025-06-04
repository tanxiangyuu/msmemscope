// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "tensor_dumper.h"

#include <fstream>
#include <vector>
#include "utils.h"
#include "kernel_hooks/acl_hooks.h"
#include "file.h"
#include "event_report.h"
#include "calculate_md5.h"
#include "tensor_monitor.h"
#include "op_excute_watch.h"

namespace Leaks {
void CleanFileName(std::string& fileName)
{
    for (size_t i = 0; i < fileName.size(); i++) {
        if (fileName[i] == '/') {
            fileName[i] = '.';
        }
    }
    return;
}

TensorDumper::TensorDumper()
{
    Config config = EventReport::Instance(CommType::SOCKET).GetConfig();
    fullContent_ = config.watchConfig.fullContent;

    dumpDir_ = std::string(config.outputDir) + "/watch_dump";

    if (!IsDumpFullContent()) {
        int32_t devId = GD_INVALID_NUM;
        if (GetDevice(&devId) == RT_ERROR_INVALID_VALUE || devId == GD_INVALID_NUM) {
            CLIENT_ERROR_LOG("Get device id failed, " + std::to_string(devId));
        }
        fileName_ = "watch_dump_md5_" + std::to_string(devId) + "_";
    }
}

TensorDumper::~TensorDumper()
{
    if (csvFile_ != nullptr) {
        fclose(csvFile_);
        csvFile_ = nullptr;
    }
}

bool TensorDumper::IsDumpFullContent()
{
    return fullContent_;
}

bool TensorDumper::DumpTensorBinary(const std::vector<char> &hostData, std::string& fileName)
{
    CleanFileName(fileName);

    Utility::UmaskGuard guard{Utility::DEFAULT_UMASK_FOR_BIN_FILE};
    std::ofstream outFile(dumpDir_ + "/" + fileName, std::ios::binary);
    if (!outFile) {
        return false;
    }

    outFile.write(hostData.data(), hostData.size());

    if (!outFile.good()) {
        return false;
    }

    outFile.close();

    return true;
}

bool TensorDumper::DumpTensorMD5(const std::vector<char> &hostData, std::string& fileName)
{
    auto MD5Value = GetTensorMD5(hostData);
    std::lock_guard<std::mutex> lock(mutex_);
    if (!Utility::CreateCsvFile(&csvFile_, dumpDir_, fileName_, WATCH_HASH_HEADERS)) {
        CLIENT_ERROR_LOG("Create csv file failed.");
    }
    if (!Utility::Fprintf(csvFile_, "%s,%s\n", fileName.c_str(), MD5Value.c_str())) {
        CLIENT_ERROR_LOG("Write tensor md5 info failed.");
        return false;
    }
    return true;
}

bool TensorDumper::DumpOneTensor(const MonitoredTensor& tensor, std::string& fileName)
{
    if (!Utility::MakeDir(dumpDir_)) {
        CLIENT_ERROR_LOG("Make dir failed.");
    }
    using AclrtMemcpy = decltype(&aclrtMemcpy);
    auto vallina = VallinaSymbol<AclLibLoader>::Instance().Get<AclrtMemcpy>("aclrtMemcpy");
    if (vallina == nullptr) {
        return false;
    }

    std::vector<char> hostData(tensor.dataSize);
    aclError ret = vallina(hostData.data(), tensor.dataSize, tensor.data, tensor.dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        return false;
    }

    if (IsDumpFullContent()) {
        return DumpTensorBinary(hostData, fileName);
    }

    return DumpTensorMD5(hostData, fileName);
}

std::string GetFileName(const std::string &op, OpEventType eventType, std::string wathcedOpName, uint64_t index)
{
    std::string type;
    if (eventType == OpEventType::ATB_START || eventType == OpEventType::ATEN_START) {
        type = "before";
    }
    if (eventType == OpEventType::ATB_END || eventType == OpEventType::ATEN_END) {
        type = "after";
    }
    std::string name = op + "-" + wathcedOpName + "_" + std::to_string(index) + "_" + type + ".bin";
    return name;
}

void TensorDumper::Dump(const std::string &op, OpEventType eventType)
{
    std::unordered_map<uint64_t, MonitoredTensor> &tensorsMap = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    uint64_t index = 0;
    for (const auto& tensorPair : tensorsMap) {
        std::string watchedOpName = OpExcuteWatch::GetInstance().GetWatchedOpName();
        auto fileName = GetFileName(op, eventType, watchedOpName, index);
        auto result = DumpOneTensor(tensorPair.second, fileName);
        if (!result) {
            LOG_WARN("Dump tensor failed, current op: %s, watched op: %s", op, watchedOpName);
        }
        ++index;
    }
    index = 0;
    std::unordered_map<uint64_t, MonitoredTensor> &tensors = TensorMonitor::GetInstance().GetPythonWatchedTensorsMap();
    for (const auto& tensorPair : tensors) {
        uint64_t ptr = static_cast<uint64_t>((std::uintptr_t)(tensorPair.second.data));
        std::string watchedOpName = GetDumpName(ptr);
        auto fileName = GetFileName(op, eventType, watchedOpName, index);
        auto dumpNums = GetDumpNums(ptr);
        if (dumpNums == 0) {
            break;
        }
        auto result = DumpOneTensor(tensorPair.second, fileName);
        if (!result) {
            LOG_WARN("Dump tensor failed, current op: %s, watched op: %s", op, watchedOpName);
        }
        if (dumpNums > 0) {
            SetDumpNums(ptr, dumpNums-1);
        }
    }
    return;
}

void TensorDumper::SetDumpNums(uint64_t ptr, int32_t dumpNums)
{
    std::lock_guard<std::mutex> lock(mapMutex_);
    auto it = dumpNumsMap_.find(ptr);
    if (it != dumpNumsMap_.end()) {
        dumpNumsMap_[ptr] = dumpNums;
    } else {
        dumpNumsMap_.insert({ptr, dumpNums});
    }
}

int32_t TensorDumper::GetDumpNums(uint64_t ptr)
{
    auto it = dumpNumsMap_.find(ptr);
    if (it != dumpNumsMap_.end()) {
        return dumpNumsMap_[ptr];
    }
    return -1;
}

void TensorDumper::DeleteDumpNums(uint64_t ptr)
{
    std::lock_guard<std::mutex> lock(mapMutex_);
    auto it = dumpNumsMap_.find(ptr);
    if (it != dumpNumsMap_.end()) {
        dumpNumsMap_.erase(ptr);
    }
}

void TensorDumper::SetDumpName(uint64_t ptr, std::string name)
{
    std::lock_guard<std::mutex> lock(mapMutex_);
    auto it = dumpNameMap_.find(ptr);
    if (it != dumpNameMap_.end()) {
        dumpNameMap_[ptr] = name;
    } else {
        dumpNameMap_.insert({ptr, name});
    }
}

std::string TensorDumper::GetDumpName(uint64_t ptr)
{
    auto it = dumpNameMap_.find(ptr);
    if (it != dumpNameMap_.end()) {
        return dumpNameMap_[ptr];
    }
    return "UNKNWON";
}

void TensorDumper::DeleteDumpName(uint64_t ptr)
{
    std::lock_guard<std::mutex> lock(mapMutex_);
    auto it = dumpNameMap_.find(ptr);
    if (it != dumpNameMap_.end()) {
        dumpNameMap_.erase(ptr);
    }
}

}