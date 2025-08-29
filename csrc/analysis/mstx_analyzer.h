// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef MSTX_ANALYZER_H
#define MSTX_ANALYZER_H

#include <unordered_map>
#include <mutex>
#include <functional>
#include "record_info.h"
#include "comm_def.h"

namespace Leaks {
/*
 * MstxAnalyzer类主要功能：
 * 1. 注册观察者，提醒观察者
 * 2. 标识打点信息
*/

using DeviceId = int32_t;
using MstxEventCallBackFunc = std::function<void(const MstxRecord&)>;

enum class MstxEventSubscriber : uint8_t {
    STEP_INNER_ANALYZER = 0,
};

class MstxAnalyzer {
public:
    static MstxAnalyzer& Instance();
    bool RecordMstx(const ClientId &clientId, const MstxRecord &mstxRecord);
    void Subscribe(const MstxEventSubscriber &subscriber, const MstxEventCallBackFunc &func);
    void UnSubscribe(const MstxEventSubscriber &subscriber);
private:
    MstxAnalyzer() = default;
    ~MstxAnalyzer() = default;

    MstxAnalyzer(const MstxAnalyzer&) = delete;
    MstxAnalyzer& operator=(const MstxAnalyzer&) = delete;
    MstxAnalyzer(MstxAnalyzer&&) = delete;
    MstxAnalyzer& operator=(MstxAnalyzer&&) = delete;

    void Notify(const MstxRecord &mstxRecord);
    std::mutex mstxMutex_;
    std::unordered_map<MstxEventSubscriber, MstxEventCallBackFunc> subscriberList_;
};

}

#endif