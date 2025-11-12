// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "dump.h"

#include "event_dispatcher.h"
#include "memory_state_manager.h"
#include "constant.h"

namespace Leaks {

Dump& Dump::GetInstance(Config config)
{
    static Dump dump{config};
    return dump;
}

Dump::Dump(Config config)
{
    config_ = config;
    auto func = std::bind(&Dump::EventHandle, this, std::placeholders::_1, std::placeholders::_2);
    std::vector<EventBaseType> eventList{
        EventBaseType::FREE,
        EventBaseType::MSTX,
        EventBaseType::OP_LAUNCH,
        EventBaseType::KERNEL_LAUNCH,
        EventBaseType::SYSTEM,
        EventBaseType::CLEAN_UP};
    EventDispatcher::GetInstance().Subscribe(SubscriberId::DUMP, eventList, EventDispatcher::Priority::Lowest, func);
    return;
}

void Dump::EventHandle(std::shared_ptr<EventBase>& event, MemoryState* state)
{
    switch (event->eventType) {
        case EventBaseType::FREE:
            if (state) {
                DumpMemoryState(state);
            }
            break;
        case EventBaseType::MSTX:
            if (auto mstxEvent = std::dynamic_pointer_cast<MstxEvent>(event)) {
                DumpMstxEvent(mstxEvent);
            }
            break;
        case EventBaseType::OP_LAUNCH:
            if (auto opLaunchEvent = std::dynamic_pointer_cast<OpLaunchEvent>(event)) {
                DumpOpLaunchEvent(opLaunchEvent);
            }
            break;
        case EventBaseType::KERNEL_LAUNCH:
            if (auto kernelLaunchEvent = std::dynamic_pointer_cast<KernelLaunchEvent>(event)) {
                DumpKernelLaunchEvent(kernelLaunchEvent);
            }
            break;
        case EventBaseType::SYSTEM:
            if (auto systemEvent = std::dynamic_pointer_cast<SystemEvent>(event)) {
                DumpSystemEvent(systemEvent);
            }
            break;
        case EventBaseType::CLEAN_UP:
            if (state) {
                DumpMemoryState(state);
            }
            break;
        default:
            break;
    }
}

void Dump::DumpMemoryState(MemoryState* state)
{
    for (auto it = state->events.begin(); it != state->events.end(); it++) {
        DumpMemoryEvent(*it, state);
    }
}

void Dump::DumpMemoryEvent(std::shared_ptr<MemoryEvent>& event, MemoryState* state)
{
    // 组装attr
    std::string attr;
    attr += "allocation_id:" + std::to_string(state->allocationId) + ",";
    attr += "addr:" + Uint64ToHexString(event->addr) + ",";
    attr += "size:" + std::to_string(event->size) + ",";
    if (event->eventType != EventBaseType::ACCESS
        && event->eventSubType != EventSubType::HAL
        && event->eventSubType != EventSubType::HOST) {
        attr += "total:" + std::to_string(event->total) + ",";
        attr += "used:" + std::to_string(event->used) + ",";
    }
    if (event->eventType == EventBaseType::ACCESS) {
        if (PoolTypeMap.find(event->poolType) != PoolTypeMap.end()) {
            attr += "type:" + PoolTypeMap.at(event->poolType) + ",";
        }
        attr += event->attr + ",";
    }
    if (event->eventType == EventBaseType::MALLOC
        && !(state->leaksDefinedOwner.empty() && state->userDefinedOwner.empty())) {
        attr += "owner:" + state->leaksDefinedOwner + state->userDefinedOwner + ",";
    }
    if (event->eventType == EventBaseType::MALLOC && !state->inefficientType.empty()) {
        attr += "inefficient_type:" + state->inefficientType + ",";
    }
    if (!attr.empty() && attr.back() == ',') {
        attr.pop_back();
    }
    event->attr = "\"{" + attr + "}\"";

    WriteToFile(event);
}

void Dump::DumpMstxEvent(std::shared_ptr<MstxEvent>& event)
{
    WriteToFile(event);
}

void Dump::DumpOpLaunchEvent(std::shared_ptr<OpLaunchEvent>& event)
{
    // 组装attr
    if (!event->attr.empty()) {
        event->attr = "\"{" + event->attr + "}\"";
    }

    WriteToFile(event);
}

void Dump::DumpKernelLaunchEvent(std::shared_ptr<KernelLaunchEvent>& event)
{
    // 组装attr
    std::string attr;
    if (event->eventSubType == EventSubType::ATB_KERNEL_START || event->eventSubType == EventSubType::ATB_KERNEL_END) {
        attr = event->attr;
    } else {
        attr += "streamId:" + event->streamId + ",";
        attr += "taskId:" + event->taskId;
    }
    event->attr = "\"{" + attr + "}\"";

    WriteToFile(event);
}

void Dump::DumpSystemEvent(std::shared_ptr<SystemEvent>& event)
{
    WriteToFile(event);
}

void Dump::WriteToFile(const std::shared_ptr<EventBase>& event)
{
    if (event->device == "N/A") {
        sharedEventLists_.push_back(event);
        return ;
    }
    auto it = handlerMap_.find(event->device);
    if (it == handlerMap_.end()) {
        handlerMap_.insert({event->device, MakeDataHandler(config_, DataType::LEAKS_EVENT, event->device)});
    }
    handlerMap_[event->device]->Write(event);
}

Dump::~Dump()
{
    for (auto& event : sharedEventLists_) {
        for (auto& handler : handlerMap_) {
            handler.second->Write(event);
        }
    }
}

}