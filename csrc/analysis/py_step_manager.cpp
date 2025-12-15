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