/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

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

namespace MemScope {

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
