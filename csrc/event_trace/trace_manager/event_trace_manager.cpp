// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include <string>
#include "event_trace_manager.h"
#include "event_report.h"
#include "cpython.h"
#include "bit_field.h"

namespace Leaks {

bool EventTraceManager::IsNeedTrace()
{
    return status_ == EventTraceStatus::IN_TRACING;
}

void EventTraceManager::InitTraceStatus()
{
    auto config = EventReport::Instance(CommType::SOCKET).GetConfig();
    auto status = (config.collectMode == static_cast<uint8_t>(CollectMode::FULL)) ? EventTraceStatus::IN_TRACING :
        EventTraceStatus::NOT_IN_TRACING;
    SetTraceStatus(status);
    return;
}

void EventTraceManager::SetTraceStatus(const EventTraceStatus status)
{
    CLIENT_INFO_LOG("Set trace status to " + std::to_string(static_cast<uint8_t>(status)) + " .");

    if (!EventReport::Instance(CommType::SOCKET).ReportTraceStatus(status)) {
        CLIENT_ERROR_LOG("Report trace status failed.\n");
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (status_ != status) {
        HandleWithTraceStatusChanged(status);
    }

    status_ = status;
    return;
}

void EventTraceManager::HandleWithTraceStatusChanged(const EventTraceStatus status)
{
    Config config =  EventReport::Instance(CommType::SOCKET).GetConfig();
    BitField<decltype(config.levelType)> levelType(config.levelType);
    if ((status == EventTraceStatus::IN_TRACING) &&
        levelType.checkBit(static_cast<size_t>(LevelType::LEVEL_OP))) {
        Utility::LeaksPythonCall("msleaks.aten_collection", "enable_aten_collector");
        return;
    }

    if (status == EventTraceStatus::NOT_IN_TRACING) {
        Utility::LeaksPythonCall("msleaks.aten_collection", "disable_aten_collector");
        return;
    }

    return;
}

}