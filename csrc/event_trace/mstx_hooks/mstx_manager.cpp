// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "mstx_manager.h"
#include <cstring>
#include "securec.h"
#include "event_report.h"
#include "record_info.h"
#include "log.h"

namespace Leaks {
// 组装普通打点信息
void MstxManager::ReportMarkA(const char* msg, int32_t streamId)
{
    MstxRecord record;
    record.markType = MarkType::MARK_A;
    record.rangeId = onlyMarkId_;
    record.streamId = streamId;

    if (strncpy_s(record.markMessage, sizeof(record.markMessage), msg, sizeof(record.markMessage) - 1) != EOK) {
        Utility::LogError("strncpy_s FAILED");
    }
    record.markMessage[sizeof(record.markMessage) - 1] = '\0';
    if (!EventReport::Instance(CommType::SOCKET).ReportMark(record)) {
        Utility::LogError("Report FAILED");
    }
}

// 组装Range开始打点信息
uint64_t MstxManager::ReportRangeStart(const char* msg, int32_t streamId)
{
    MstxRecord record;
    record.markType = MarkType::RANGE_START_A;
    record.streamId = streamId;
    if (strncpy_s(record.markMessage, sizeof(record.markMessage), msg, sizeof(record.markMessage) - 1) != EOK) {
        Utility::LogError("strncpy_s FAILED");
    }
    record.markMessage[sizeof(record.markMessage) - 1] = '\0';
    record.rangeId = GetRangeId();
    if (!EventReport::Instance(CommType::SOCKET).ReportMark(record)) {
        Utility::LogError("Report FAILED");
    }
    return record.rangeId;
}

// 组装Range结束打点信息
void MstxManager::ReportRangeEnd(uint64_t id)
{
    MstxRecord record;
    record.markType = MarkType::RANGE_END;
    record.rangeId = id;
    record.streamId = -1;
    std::string msg = "Range end from id " + std::to_string(id);
    if (strncpy_s(record.markMessage, sizeof(record.markMessage), msg.c_str(), sizeof(record.markMessage) - 1) != EOK) {
        Utility::LogError("strncpy_s FAILED");
    }
    record.markMessage[sizeof(record.markMessage) - 1] = '\0';
    if (!EventReport::Instance(CommType::SOCKET).ReportMark(record)) {
        Utility::LogError("Report FAILED");
    }
}

uint64_t MstxManager::GetRangeId()
{
    return rangeId_++;
}

}