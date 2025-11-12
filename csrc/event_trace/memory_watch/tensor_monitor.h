// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef TENSOR_MONITOR_H
#define TENSOR_MONITOR_H

#include <unordered_map>
#include <vector>
#include <mutex>

#include "log.h"

using TENSOR_ADDR = uint64_t;
using TENSOR_SIZE = uint64_t;

struct MonitoredTensor {
    void* data;
    uint64_t dataSize;
};

namespace Leaks {

class TensorMonitor {
public:
    static TensorMonitor& GetInstance()
    {
        static TensorMonitor instance;
        return instance;
    }
    void AddWatchTensor(MonitoredTensor& tensorInfo);
    void AddWatchTensor(const std::vector<MonitoredTensor>& tensorInfoLists, uint32_t outputId);
    std::unordered_map<uint64_t, MonitoredTensor> GetCmdWatchedTensorsMap();
    uint32_t GetCmdWatchedOutputId();
    std::unordered_map<uint64_t, MonitoredTensor> GetPythonWatchedTensorsMap();
    void DeleteWatchTensor(MonitoredTensor& tensorInfo);
    void ClearCmdWatchTensor();
    bool IsInMonitoring();
private:
    TensorMonitor() = default;
    ~TensorMonitor() = default;
    TensorMonitor(const TensorMonitor&) = delete;
    TensorMonitor& operator=(const TensorMonitor&) = delete;
private:
    uint32_t outputId_ = UINT32_MAX;
    std::unordered_map<uint64_t, MonitoredTensor> cmdWatchedTensorsMap_ = {};
    std::unordered_map<uint64_t, MonitoredTensor> pythonWatchedTensorsMap_ = {};
    std::mutex mapMutex_;
};

}

#endif
