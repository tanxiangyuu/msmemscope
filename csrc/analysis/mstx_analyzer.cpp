// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "mstx_analyzer.h"

namespace Leaks {

MstxAnalyzer& MstxAnalyzer::Instance()
{
    static MstxAnalyzer instance;
    return instance;
}

void MstxAnalyzer::RegisterAnalyzer(std::shared_ptr<AnalyzerBase> analyzer)
{
    analyzerList.push_back(analyzer);
}

void MstxAnalyzer::UnregisterAnalyzer(std::shared_ptr<AnalyzerBase> analyzer)
{
    analyzerList.remove(analyzer);
}

void MstxAnalyzer::Notify(const DeviceId &deviceId, const uint64_t &stepId, const MstxRecord &mstxRecord)
{
    for (std::shared_ptr<AnalyzerBase> analyzer : analyzerList) {
        analyzer->ReceiveMstxMsg(deviceId, stepId, mstxRecord);
    }
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
        Notify(deviceId, stepId, mstxRecord);
        return true;
    } else if (mstxRecord.markType == MarkType::RANGE_END) {
        Utility::LogInfo("[npu %ld][client %u][stepid %llu][streamid %d][end]: %s",
            deviceId,
            clientId,
            stepId,
            mstxRecord.streamId,
            mstxRecord.markMessage);
        Notify(deviceId, stepId, mstxRecord);
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