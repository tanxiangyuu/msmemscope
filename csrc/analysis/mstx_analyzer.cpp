// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "mstx_analyzer.h"
#include "utility/log.h"
#include "utility/ustring.h"

namespace MemScope {

MstxAnalyzer& MstxAnalyzer::Instance()
{
    static MstxAnalyzer instance;
    return instance;
}

void MstxAnalyzer::Subscribe(const MstxEventSubscriber &subscriber, const MstxEventCallBackFunc &func)
{
    if (subscriberList_.find(subscriber) != subscriberList_.end()) {
        LOG_ERROR("Add elements repeatedly, subscriber : %u", static_cast<uint8_t>(subscriber));
        return;
    }

    subscriberList_.insert({subscriber, func});
    return;
}

void MstxAnalyzer::UnSubscribe(const MstxEventSubscriber &subscriber)
{
    if (subscriberList_.find(subscriber) == subscriberList_.end()) {
        LOG_ERROR("Cannot delete elements, subscriber : %u", static_cast<uint8_t>(subscriber));
        return;
    }

    subscriberList_.erase(subscriber);
    return;
}

void MstxAnalyzer::Notify(const MstxRecord &mstxRecord)
{
    std::lock_guard<std::mutex> lock(mstxMutex_);
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
    const TLVBlock* tlv = GetTlvBlock(mstxRecord, TLVBlockType::MARK_MESSAGE);
    std::string markMessage = tlv == nullptr ? "" : tlv->data;
    Utility::ToSafeString(markMessage);
    if (mstxRecord.markType == MarkType::RANGE_START_A) {
        LOG_INFO("[npu %ld][client %u][stepid %llu][streamid %d][start]: %s",
            deviceId,
            clientId,
            stepId,
            mstxRecord.streamId,
            markMessage.c_str());
        Notify(mstxRecord);
        return true;
    } else if (mstxRecord.markType == MarkType::RANGE_END) {
        LOG_INFO("[npu %ld][client %u][stepid %llu][streamid %d][end]: %s",
            deviceId,
            clientId,
            stepId,
            mstxRecord.streamId,
            markMessage.c_str());
        Notify(mstxRecord);
        return true;
    } else if (mstxRecord.markType == MarkType::MARK_A) {
        LOG_INFO("[npu %ld][client %u][stepid %llu][streamid %d][mark]: %s",
            deviceId,
            clientId,
            stepId,
            mstxRecord.streamId,
            markMessage.c_str());
        return true;
    }
    return false;
}

}