// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#ifndef ATEN_MANAGER_H
#define ATEN_MANAGER_H

#include <cstdint>
#include <iostream>
#include "utils.h"
#include "event_report.h"
#include "bit_field.h"
#include "op_watch/tensor_monitor.h"
#include "op_watch/op_excute_watch.h"

namespace Leaks {

class AtenManager {
public:
    static AtenManager& GetInstance();
    void ProcessMsg(const char* msg, int32_t streamId);

private:
    AtenManager() = default;
    ~AtenManager() = default;
    AtenManager(const AtenManager&) = delete;
    AtenManager& operator=(const AtenManager&) = delete;
    AtenManager(AtenManager&& other) = delete;
    AtenManager& operator=(AtenManager&& other) = delete;

    bool ExtractTensorInfo(const char* msg, const std::string &key, std::string &value);
    void ReportAtenLaunch(const char* msg, int32_t streamId, bool isAtenBegin);
    void ReportAtenAccess(const char* msg, int32_t streamId);
    bool IsAtenLaunchEnable();
    bool IsAtenAccessEnable();
    bool IsWatchEnable();
    void ParseAtenAccessMsg(const char* msg, MemAccessRecord &record, std::string &dtype,
        std::string &shape, std::string &isOutput);
private:
    std::vector<MonitoredTensor> outputTensors_ = {};
};

}

#endif
