// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef MSTX_ANALYZER_H
#define MSTX_ANALYZER_H

#include <list>
#include "host_injection/core/LocalProcess.h"
#include "analyzer_base.h"

namespace Leaks {
/*
 * MstxAnalyzer类主要功能：
 * 1. 注册观察者，提醒观察者
 * 2. 标识打点信息
*/


using DeviceId = int32_t;

class MstxAnalyzer {
public:
    MstxAnalyzer() = default;
    void RecordMstx(const ClientId &clientId, const MstxRecord &mstxRecord);
    void RegisterAnalyzer(std::shared_ptr<AnalyzerBase> analyzer);
    void UnregisterAnalyzer(std::shared_ptr<AnalyzerBase> analyzer);
    ~MstxAnalyzer() = default;
private:
    std::list<std::shared_ptr<AnalyzerBase>> analyzerList;
    void Notify(const DeviceId &deviceId, const uint64_t &rangeId, const MstxRecord &mstxRecord);
};

}

#endif