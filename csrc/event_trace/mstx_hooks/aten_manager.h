// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#ifndef ATEN_MANAGER_H
#define ATEN_MANAGER_H

#include <cstdint>
#include <iostream>
#include "utils.h"
#include "event_report.h"
#include "bit_field.h"
#include "memory_watch/tensor_monitor.h"
#include "memory_watch/memory_watch.h"

namespace Leaks {

struct AtenAccessTensorInfo {
    std::string addr;
    std::string dtype;
    std::string shape;
    std::string size;
    std::string name;
    std::string isWrite;
    std::string isRead;
    std::string isOutput;
};

class AtenManager {
public:
    static AtenManager& GetInstance();
    void ProcessMsg(const char* msg, int32_t streamId);
    AtenManager();
private:
    ~AtenManager() = default;
    AtenManager(const AtenManager&) = delete;
    AtenManager& operator=(const AtenManager&) = delete;
    AtenManager(AtenManager&& other) = delete;
    AtenManager& operator=(AtenManager&& other) = delete;

    bool ExtractTensorInfo(const char* msg, const std::string &key, std::string &value);
    void ExtractTensorFields(const char* msg, AtenAccessTensorInfo& info);
    void ReportAtenLaunch(const char* msg, int32_t streamId, bool isAtenBegin);
    void ReportAtenAccess(const char* msg, int32_t streamId);
    bool IsFirstWatchedOp(const char* name);
    bool IsLastWatchedOp(const char* name);
private:
    std::vector<MonitoredTensor> outputTensors_ = {};
    bool isWatchEnable_ = false;
    bool isfirstWatchOpSet_ = false;
    std::string firstWatchOp_ = {};
    std::string lastWatchOp_ = {};
};

}

#endif
