// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef PY_STEP_MANAGER_H
#define PY_STEP_MANAGER_H

#include <unordered_map>
#include <mutex>
#include <functional>
#include "record_info.h"
#include "comm_def.h"

namespace MemScope {
/*
 * PyStepManager类主要功能：
 * 1. 注册观察者，提醒观察者
 * 2. 标识打点信息
*/

using DeviceId = int32_t;
using PyStepEventCallBackFunc = std::function<void(const PyStepRecord&)>;

enum class PyStepEventSubscriber : uint8_t {
    STEP_INNER_ANALYZER = 0,
};

class PyStepManager {
public:
    static PyStepManager& Instance();
    void RecordPyStep(const ClientId &clientId, const PyStepRecord &PyStepRecord);
    void Subscribe(const PyStepEventSubscriber &subscriber, const PyStepEventCallBackFunc &func);
    void UnSubscribe(const PyStepEventSubscriber &subscriber);
private:
    PyStepManager() = default;
    ~PyStepManager() = default;

    PyStepManager(const PyStepManager&) = delete;
    PyStepManager& operator=(const PyStepManager&) = delete;
    PyStepManager(PyStepManager&&) = delete;
    PyStepManager& operator=(PyStepManager&&) = delete;

    void Notify(const PyStepRecord &PyStepRecord);
    std::mutex pyStepMutex_;
    std::unordered_map<PyStepEventSubscriber, PyStepEventCallBackFunc> subscriberList_;
};

}

#endif