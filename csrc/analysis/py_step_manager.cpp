// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "py_step_manager.h"
#include "utility/log.h"
#include "utility/ustring.h"

namespace MemScope {

PyStepManager& PyStepManager::Instance()
{
    static PyStepManager instance;
    return instance;
}

void PyStepManager::Subscribe(const PyStepEventSubscriber &subscriber, const PyStepEventCallBackFunc &func)
{
    if (subscriberList_.find(subscriber) != subscriberList_.end()) {
        LOG_ERROR("Add elements repeatedly, subscriber : %u", static_cast<uint8_t>(subscriber));
        return;
    }

    subscriberList_.insert({subscriber, func});
    return;
}

void PyStepManager::UnSubscribe(const PyStepEventSubscriber &subscriber)
{
    if (subscriberList_.find(subscriber) == subscriberList_.end()) {
        LOG_ERROR("Cannot delete elements, subscriber : %u", static_cast<uint8_t>(subscriber));
        return;
    }

    subscriberList_.erase(subscriber);
    return;
}

void PyStepManager::Notify(const PyStepRecord &pyStepRecord)
{
    std::lock_guard<std::mutex> lock(pyStepMutex_);
    for (auto &subscriber : subscriberList_) {
        if (subscriber.second != nullptr) {
            subscriber.second(pyStepRecord);
        }
    }
    
    return;
}

void PyStepManager::RecordPyStep(const ClientId &clientId, const PyStepRecord &pyStepRecord)
{
    DeviceId deviceId = pyStepRecord.devId;
    uint64_t stepId = pyStepRecord.stepId;

    LOG_INFO("[npu %ld][client %u]: msleaks.step(): Now in step: %llu",
        deviceId,
        clientId,
        stepId);
    Notify(pyStepRecord);

    return;
}

}