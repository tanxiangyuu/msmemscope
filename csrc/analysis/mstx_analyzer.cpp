// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "mstx_analyzer.h"
#include "utility/log.h"

namespace Leaks {

MstxAnalyzer& MstxAnalyzer::Instance()
{
    static MstxAnalyzer instance;
    return instance;
}

void MstxAnalyzer::Subscribe(const MstxEventSubscriber &subscriber, const MstxEventCallBackFunc &func)
{
    if (subscriberList_.find(subscriber) != subscriberList_.end()) {
        Utility::LogError("Add elements repeatedly, subscriber : %u", static_cast<uint8_t>(subscriber));
        return;
    }

    subscriberList_.insert({subscriber, func});
    return;
}

void MstxAnalyzer::UnSubscribe(const MstxEventSubscriber &subscriber)
{
    if (subscriberList_.find(subscriber) == subscriberList_.end()) {
        Utility::LogError("Cannot delete elements, subscriber : %u", static_cast<uint8_t>(subscriber));
        return;
    }

    subscriberList_.erase(subscriber);
    return;
}

void MstxAnalyzer::Notify(const MstxRecord &mstxRecord)
{
    for (auto &subscriber : subscriberList_) {
        if (subscriber.second != nullptr) {
            subscriber.second(mstxRecord);
        }
    }
    
    return;
}

bool MstxAnalyzer::RecordMstx(const ClientId &clientId, const MstxRecord &mstxRecord)
{
    DeviceId deviceId = mstxRecord.devId;
    uint64_t stepId = mstxRecord.stepId;
    if (mstxRecord.markType == MarkType::RANGE_START_A) {
        Utility::LogInfo("[npu %ld][client %u][stepid %llu][streamid %d][start]: %s",
            deviceId,
            clientId,
            stepId,
            mstxRecord.streamId,
            mstxRecord.markMessage);
        Notify(mstxRecord);
        return true;
    } else if (mstxRecord.markType == MarkType::RANGE_END) {
        Utility::LogInfo("[npu %ld][client %u][stepid %llu][streamid %d][end]: %s",
            deviceId,
            clientId,
            stepId,
            mstxRecord.streamId,
            mstxRecord.markMessage);
        Notify(mstxRecord);
        return true;
    } else if (mstxRecord.markType == MarkType::MARK_A) {
        Utility::LogInfo("[npu %ld][client %u][stepid %llu][streamid %d][mark]: %s",
            deviceId,
            clientId,
            stepId,
            mstxRecord.streamId,
            mstxRecord.markMessage);
        return true;
    }
    return false;
}

}