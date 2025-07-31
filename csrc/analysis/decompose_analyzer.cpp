// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "decompose_analyzer.h"

#include <string>

#include "event_dispatcher.h"
#include "constant.h"

namespace Leaks {

const std::string DecomposeAnalyzer::cannStr = "CANN";
const std::string DecomposeAnalyzer::ptaStr = "PTA";
const std::string DecomposeAnalyzer::ptaWorkspaceStr = "PTA_WORKSPACE";
const std::string DecomposeAnalyzer::atbStr = "ATB";
const std::string DecomposeAnalyzer::mindsporeStr = "MINDSPORE";
const size_t DecomposeAnalyzer::ptaStrLen = DecomposeAnalyzer::ptaStr.length();

const std::string DecomposeAnalyzer::atenStr = "@ops@aten";

DecomposeAnalyzer& DecomposeAnalyzer::GetInstance()
{
    static DecomposeAnalyzer analyzer{};
    return analyzer;
}

DecomposeAnalyzer::DecomposeAnalyzer()
{
    auto func = std::bind(&DecomposeAnalyzer::EventHandle, this, std::placeholders::_1, std::placeholders::_2);
    std::vector<EventBaseType> eventList{
        EventBaseType::MALLOC,
        EventBaseType::ACCESS,
        EventBaseType::MEMORY_OWNER};
    EventDispatcher::GetInstance().Subscribe(
        SubscriberId::DECOMPOSE_ANALYZER, eventList, EventDispatcher::Priority::High, func);
}

void DecomposeAnalyzer::EventHandle(std::shared_ptr<EventBase>& event, MemoryState* state)
{
    if (event->eventType == EventBaseType::MALLOC) {
        auto memEvent = std::dynamic_pointer_cast<MemoryEvent>(event);
        if (memEvent != nullptr && state != nullptr) {
            InitOwner(memEvent, state);
        }
    } else if (event->eventType == EventBaseType::ACCESS) {
        auto memEvent = std::dynamic_pointer_cast<MemoryEvent>(event);
        if (memEvent != nullptr && state != nullptr) {
            UpdateOwnerByAtenAccess(memEvent, state);
        }
    } else if (event->eventType == EventBaseType::MEMORY_OWNER) {
        auto memOwnerEvent = std::dynamic_pointer_cast<MemoryOwnerEvent>(event);
        if (memOwnerEvent != nullptr && state != nullptr) {
            UpdateOwner(memOwnerEvent, state);
        }
    }
}

void DecomposeAnalyzer::InitOwner(std::shared_ptr<MemoryEvent>& event, MemoryState* state)
{
    switch (event->eventSubType) {
        case EventSubType::HAL: {
            auto it = MODULE_HASH_TABLE.find(event->moduleId);
            if (it != MODULE_HASH_TABLE.end()) {
                state->leaksDefinedOwner = cannStr + "@" + it->second;
            } else {
                state->leaksDefinedOwner = cannStr + "@UNKNOWN";
            }
            state->userDefinedOwner = event->describeOwner;
            break;
        }
        case EventSubType::PTA_CACHING: {
            state->leaksDefinedOwner = ptaStr;
            state->userDefinedOwner = event->describeOwner;
            break;
        }
        case EventSubType::PTA_WORKSPACE: {
            state->leaksDefinedOwner = ptaWorkspaceStr;
            state->userDefinedOwner = event->describeOwner;
            break;
        }
        case EventSubType::MINDSPORE: {
            state->leaksDefinedOwner = mindsporeStr;
            state->userDefinedOwner = event->describeOwner;
            break;
        }
        case EventSubType::ATB: {
            state->leaksDefinedOwner = atbStr;
            state->userDefinedOwner = event->describeOwner;
            break;
        }
        default:
            break;
    }
}

void DecomposeAnalyzer::UpdateOwnerByAtenAccess(std::shared_ptr<MemoryEvent>& event, MemoryState* state)
{
    if (event->eventSubType != EventSubType::ATEN_READ
        && event->eventSubType != EventSubType::ATEN_WRITE
        && event->eventSubType != EventSubType::ATEN_READ_OR_WRITE) {
        return;
    }

    if (state->leaksDefinedOwner.rfind(ptaStr, 0) != 0) {
        return;
    }

    if (state->leaksDefinedOwner.length() == ptaStrLen) {
        state->leaksDefinedOwner += atenStr;
    }
}

void DecomposeAnalyzer::UpdateOwner(std::shared_ptr<MemoryOwnerEvent>& event, MemoryState* state)
{
    if (event->eventSubType == EventSubType::DESCRIBE_OWNER && !(event->owner).empty()) {
        state->userDefinedOwner += event->owner;
    } else if (event->eventSubType == EventSubType::TORCH_OPTIMIZER_STEP_OWNER && !(event->owner).empty()) {
        if (state->leaksDefinedOwner.rfind(ptaStr, 0) != 0) {
            return;
        }

        if (state->leaksDefinedOwner.length() == ptaStrLen) {
            state->leaksDefinedOwner += event->owner;
        } else if (event->owner != atenStr) {
            // 部分内存有可能先作为算子操作的内容，然后被识别为其他类型，如weight，
            // 则优先用weight覆盖aten，而aten不能覆盖其他类型
            state->leaksDefinedOwner = ptaStr + event->owner;
        }
    }
}
}