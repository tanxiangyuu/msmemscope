// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "tensor_dumper.h"

#include <fstream>
#include <vector>
#include "utils.h"
#include "file.h"
#include "event_report.h"
#include "calculate_data_check_sum.h"
#include "tensor_monitor.h"
#include "op_excute_watch.h"
#include "client_process.h"

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
    fullContent_ = GetConfig().watchConfig.fullContent;

    dumpDir_ = std::string(GetConfig().outputDir) + "/watch_dump";
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

bool TensorDumper::DumpTensorBinary(const std::vector<uint8_t> &hostData, std::string& fileName)
{
    CleanFileName(fileName);

    Utility::UmaskGuard guard{Utility::DEFAULT_UMASK_FOR_BIN_FILE};
    std::ofstream outFile(dumpDir_ + "/" + fileName, std::ios::binary);
    if (!outFile) {
        return false;
    }

    outFile.write(reinterpret_cast<const char*>(hostData.data()), hostData.size());

    if (!outFile.good()) {
        return false;
    }

    outFile.close();

    return true;
}

bool TensorDumper::DumpTensorHashValue(const std::vector<uint8_t> &hostData, std::string& fileName)
{
    if (csvFile_ == nullptr) {
        int32_t devId = GD_INVALID_NUM;
        if (GetDevice(&devId) != ACL_SUCCESS || devId == GD_INVALID_NUM) {
            CLIENT_ERROR_LOG("Get device id failed, " + std::to_string(devId));
        }
        fileName_ = "watch_dump_data_check_sum_" + std::to_string(devId) + "_";
        if (!Utility::CreateCsvFile(&csvFile_, dumpDir_, fileName_, WATCH_HASH_HEADERS)) {
            CLIENT_ERROR_LOG("Create csv file failed.");
            return false;
        }
    }

    auto hashValue = CalculateDataCheckSum64(hostData);
    if (!Utility::Fprintf(csvFile_, "%s,%s\n", fileName.c_str(), hashValue.c_str())) {
        CLIENT_ERROR_LOG("Write tensor data check sum info failed.");
        return false;
    }
    return true;
}

bool TensorDumper::DumpOneTensor(const MonitoredTensor& tensor, std::string& fileName)
{
    if (!Utility::MakeDir(dumpDir_)) {
        CLIENT_ERROR_LOG("Make dir failed.");
        return false;
    }
    using AclrtMemcpy = decltype(&aclrtMemcpy);
    auto vallina = VallinaSymbol<AclLibLoader>::Instance().Get<AclrtMemcpy>("aclrtMemcpy");
    if (vallina == nullptr) {
        return false;
    }

    std::vector<uint8_t> hostData(tensor.dataSize);
    aclError ret = vallina(hostData.data(), tensor.dataSize, tensor.data, tensor.dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    if (IsDumpFullContent()) {
        return DumpTensorBinary(hostData, fileName);
    }

    return DumpTensorHashValue(hostData, fileName);
}

uint64_t TensorDumper::CountOpName(const std::string& name)
{
    std::lock_guard<std::mutex> guard(opNameMutex_);
    if (opNameCnt_.find(name) == opNameCnt_.end()) {
        opNameCnt_[name] = 1;
    } else {
        opNameCnt_[name] += 1;
    }
    return opNameCnt_[name];
}

std::string TensorDumper::GetFileName(const std::string &op, RecordSubType eventType,
    std::string wathcedOpName, uint64_t index, bool isFirstOp)
{
    std::string type;
    std::string opName = op;
    if (eventType == RecordSubType::ATB_START || eventType == RecordSubType::ATEN_START) {
        opName += "_" + std::to_string(CountOpName(op));
        type = "before";
    }
    if (eventType == RecordSubType::ATB_END || eventType == RecordSubType::ATEN_END) {
        if (isFirstOp) {
            opName += "_" + std::to_string(CountOpName(op));
        } else {
            opName += "_" + std::to_string(opNameCnt_[op]);
        }
        type = "after";
    }
    std::string name = opName + "-" + wathcedOpName + "_" + std::to_string(index) + "_" + type + ".bin";
    return name;
}

void TensorDumper::SynchronizeStream(aclrtStream stream)
{
    using AclrtSynchronizeStream = decltype(&aclrtSynchronizeStream);
    static auto vallina = VallinaSymbol<AclLibLoader>::Instance().Get<AclrtSynchronizeStream>("aclrtSynchronizeStream");
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("Gey aclrtSynchronizeStream func ptr failed");
        return;
    }
 
    int ret = vallina(stream);
    if (ret != ACL_SUCCESS) {
        CLIENT_ERROR_LOG("Dump tensor synchronize stream failed, ret is" + std::to_string(ret));
        return;
    }
    return;
}

void TensorDumper::Dump(aclrtStream stream, const std::string &op, RecordSubType eventType, bool isFirstOp)
{
    SynchronizeStream(stream);
    std::unordered_map<uint64_t, MonitoredTensor> tensorsMap = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    uint64_t index = 0;
    for (const auto& tensorPair : tensorsMap) {
        std::string watchedOpName = OpExcuteWatch::GetInstance().GetWatchedOpName();
        auto outputId = TensorMonitor::GetInstance().GetCmdWatchedOutputId();
        auto fileName = GetFileName(op, eventType, watchedOpName, outputId > 0 ? outputId : index, isFirstOp);
        auto result = DumpOneTensor(tensorPair.second, fileName);
        if (!result) {
            CLIENT_ERROR_LOG("Dump tensor failed, current op: " + op + ", watched op: " + watchedOpName);
        }
        ++index;
    }
    index = 0;
    std::unordered_map<uint64_t, MonitoredTensor> tensors = TensorMonitor::GetInstance().GetPythonWatchedTensorsMap();
    for (const auto& tensorPair : tensors) {
        uint64_t ptr = static_cast<uint64_t>((std::uintptr_t)(tensorPair.second.data));
        std::string watchedOpName = GetDumpName(ptr);
        auto fileName = GetFileName(op, eventType, watchedOpName, index, isFirstOp);
        auto dumpNums = GetDumpNums(ptr);
        if (dumpNums == 0) {
            break;
        }
        auto result = DumpOneTensor(tensorPair.second, fileName);
        if (!result) {
            CLIENT_ERROR_LOG("Dump tensor failed, current op: " + op + ", watched op: " + watchedOpName);
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