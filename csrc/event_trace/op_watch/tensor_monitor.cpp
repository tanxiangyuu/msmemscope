// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "tensor_monitor.h"
#include "client_process.h"

namespace Leaks {

void TensorMonitor::AddWatchTensor(MonitoredTensor& tensorInfo)
{
    std::lock_guard<std::mutex> lock(mapMutex_);
    uint64_t ptr = static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(tensorInfo.data));
    auto it = pythonWatchedTensorsMap_.find(ptr);
    if (it != pythonWatchedTensorsMap_.end()) {
        pythonWatchedTensorsMap_[ptr] = tensorInfo;
    } else {
        pythonWatchedTensorsMap_.insert({ptr, tensorInfo});
    }
}

void TensorMonitor::AddWatchTensor(const std::vector<MonitoredTensor>& tensorInfoLists, uint32_t outputId)
{
    std::lock_guard<std::mutex> lock(mapMutex_);
    outputId_ = outputId;
    for (auto& tensorInfo : tensorInfoLists) {
        uint64_t ptr = static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(tensorInfo.data));
        auto it = cmdWatchedTensorsMap_.find(ptr);
        if (it != cmdWatchedTensorsMap_.end()) {
            cmdWatchedTensorsMap_[ptr] = tensorInfo;
        } else {
            cmdWatchedTensorsMap_.insert({ptr, tensorInfo});
        }
    }
}

std::unordered_map<uint64_t, MonitoredTensor>& TensorMonitor::GetCmdWatchedTensorsMap()
{
    return cmdWatchedTensorsMap_;
}

uint32_t TensorMonitor::GetCmdWatchedOutputId()
{
    return outputId_;
}

std::unordered_map<uint64_t, MonitoredTensor>& TensorMonitor::GetPythonWatchedTensorsMap()
{
    return pythonWatchedTensorsMap_;
}

void TensorMonitor::DeleteWatchTensor(MonitoredTensor& tensorInfo)
{
    std::lock_guard<std::mutex> lock(mapMutex_);
    uint64_t ptr = static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(tensorInfo.data));
    auto it = pythonWatchedTensorsMap_.find(ptr);
    if (it != pythonWatchedTensorsMap_.end()) {
        pythonWatchedTensorsMap_.erase(ptr);
    } else {
        CLIENT_WARN_LOG("Failed to delete the tensor. The tensor ptr of " + std::to_string(ptr) + " is not watched.");
    }
}

void TensorMonitor::ClearCmdWatchTensor()
{
    std::lock_guard<std::mutex> lock(mapMutex_);
    cmdWatchedTensorsMap_.clear();
}

}
