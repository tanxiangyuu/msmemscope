// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef ANALYZERBASE_H
#define ANALYZERBASE_H

#include "log.h"
#include "framework/config_info.h"
#include "framework/record_info.h"
#include "host_injection/core/Communication.h"

namespace Leaks {

// AnalyzerBase类为各种分析类的基类

using DeviceId = int32_t;

class AnalyzerBase {
public:
    virtual bool Record(const ClientId &clientId, const EventRecord &record) = 0;
    virtual void ReceiveMstxMsg(const DeviceId &deviceId, const uint64_t &rangeId, const MstxRecord &mstxRecord) = 0;
};

}
#endif