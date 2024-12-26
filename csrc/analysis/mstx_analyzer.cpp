// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "mstx_analyzer.h"

namespace Leaks {

void MstxAnalyzer::RegisterAnalyzer(std::shared_ptr<AnalyzerBase> analyzer)
{
    analyzerList.push_back(analyzer);
}

void MstxAnalyzer::UnregisterAnalyzer(std::shared_ptr<AnalyzerBase> analyzer)
{
    analyzerList.remove(analyzer);
}

void MstxAnalyzer::Notify(const DeviceId &deviceId, const uint64_t &rangeId, const MstxRecord &mstxRecord)
{
    for (std::shared_ptr<AnalyzerBase> analyzer : analyzerList) {
        analyzer->ReceiveMstxMsg(deviceId, rangeId, mstxRecord);
    }
}

void MstxAnalyzer::RecordMstx(const ClientId &clientId, const MstxRecord &mstxRecord)
{
    DeviceId deviceId = mstxRecord.devId;
    uint64_t rangeId = mstxRecord.rangeId;
    if (mstxRecord.markType == MarkType::RANGE_START_A) {
        Utility::LogInfo("[npu %ld][client %u][rangeid %llu][start]: %s",
            deviceId,
            clientId,
            rangeId,
            mstxRecord.markMessage);
        Notify(deviceId, rangeId, mstxRecord);
    } else if (mstxRecord.markType == MarkType::RANGE_END) {
        Utility::LogInfo("[npu %ld][client %u][rangeid %llu][end]: %s",
            deviceId,
            clientId,
            rangeId,
            mstxRecord.markMessage);
        Notify(deviceId, rangeId, mstxRecord);
    } else if (mstxRecord.markType == MarkType::MARK_A) {
        Utility::LogInfo("[npu %ld][client %u][rangeid %llu][mark]: %s",
            deviceId,
            clientId,
            rangeId,
            mstxRecord.markMessage);
    }
    return;
}

}