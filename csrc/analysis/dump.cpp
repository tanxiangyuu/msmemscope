// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "dump.h"

#include "event_dispatcher.h"
#include "module_info.h"
#include "memory_state_manager.h"

namespace Leaks {

Dump& Dump::GetInstance(Config config)
{
    static Dump dump{config};
    return dump;
}

Dump::Dump(Config config)
{
    config_ = config;
    handler_ = MakeDataHandler(config_, DumpClass::LEAKS_RECORD);
    auto func = std::bind(&Dump::EventHandle, this, std::placeholders::_1, std::placeholders::_2);
    std::vector<EventBaseType> eventList{
        EventBaseType::MALLOC,
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
        case EventBaseType::MALLOC:
            if (state) {
                DumpMemEventBeforeMalloc(state);
            }
            break;
        case EventBaseType::FREE:
            if (state) {
                DumpMemEventAfterFree(state);
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
            if (auto cleanUpEvent = std::dynamic_pointer_cast<CleanUpEvent>(event)) {
                DumpMemEventBeforeCleanUp(cleanUpEvent);
            }
            break;
        default:
            break;
    }

    // 暂时把删除逻辑放到这里
    if (event->eventType == EventBaseType::FREE || event->eventType == EventBaseType::CLEAN_UP) {
        MemoryStateManager::GetInstance().DeteleState(event->poolType, MemoryStateKey{event->pid, event->addr});
    }
}

void Dump::DumpMemEventBeforeMalloc(MemoryState* state)
{
    if (state->events.size() > 1) {
        // dump Malloc事件前的所有事件，并删除已dump的数据
        for (auto it = state->events.begin(); it != state->events.end() - 1;) {
            DumpMemoryEvent(*it, state);
            it = state->events.erase(it); // erase返回下一个有效迭代器
        }
    }
}

void Dump::DumpMemEventAfterFree(MemoryState* state)
{
    for (auto it = state->events.begin(); it != state->events.end();) {
        DumpMemoryEvent(*it, state);
        it = state->events.erase(it); // erase返回下一个有效迭代器
    }
}

void Dump::DumpMemEventBeforeCleanUp(std::shared_ptr<CleanUpEvent>& event)
{
    auto state = MemoryStateManager::GetInstance().GetState(
        event->poolType, MemoryStateKey{event->pid, event->addr});
    if (state == nullptr) {
        return;
    }

    for (auto it = state->events.begin(); it != state->events.end();) {
        DumpMemoryEvent(*it, state);
        it = state->events.erase(it); // erase返回下一个有效迭代器
    }
}

inline DumpContainer EventToContainer(const std::shared_ptr<EventBase>& event)
{
    DumpContainer container;
    container.id = event->id;
    container.event = EVENT_BASE_TYPE_MAP.find(event->eventType) == EVENT_BASE_TYPE_MAP.end()
                    ? "N/A" : EVENT_BASE_TYPE_MAP.at(event->eventType);
    container.eventType = EVENT_SUB_TYPE_MAP.find(event->eventSubType) == EVENT_SUB_TYPE_MAP.end()
                    ? "N/A" : EVENT_SUB_TYPE_MAP.at(event->eventSubType);
    container.name = event->name;
    container.timestamp = event->timestamp;
    container.pid = event->pid;
    container.tid = event->tid;
    container.deviceId = event->device;

    return container;
}

void Dump::DumpMemoryEvent(std::shared_ptr<MemoryEvent>& event, MemoryState* state)
{
    auto container = EventToContainer(event);
    container.addr = std::to_string(event->addr);

    // 组装attr
    std::string attr;
    attr += "addr:" + std::to_string(event->addr) + ",";
    attr += "size:" + std::to_string(event->size) + ",";
    if (event->eventType != EventBaseType::ACCESS
        && event->eventSubType != EventSubType::HAL
        && event->eventSubType != EventSubType::HOST) {
        attr += "total:" + std::to_string(event->total) + ",";
        attr += "used:" + std::to_string(event->used) + ",";
    }
    if (event->eventSubType == EventSubType::HAL) {
        attr += "MID:" + std::to_string(event->moduleId) + ",";
    }
    if (event->eventType == EventBaseType::ACCESS) {
        attr += event->attr + ",";
    }
    if (event->eventType == EventBaseType::MALLOC
        && !(state->leaksDefinedOwner.empty() && state->userDefinedOwner.empty())) {
        attr += "owner:" + state->leaksDefinedOwner + state->userDefinedOwner + ",";
    }
    if (event->eventType == EventBaseType::MALLOC && !state->inefficientType.empty()) {
        attr += "inefficient type:" + state->inefficientType + ",";
    }
    if (!attr.empty() && attr.back() == ',') {
        attr.pop_back();
    }
    container.attr = "\"{" + attr + "}\"";

    std::lock_guard<std::mutex> lock(fileMutex_);
    if (handler_->Init()) {
        handler_->Write(&container, CallStackString{event->cCallStack, event->pyCallStack});
    }
}

void Dump::DumpMstxEvent(std::shared_ptr<MstxEvent>& event)
{
    auto container = EventToContainer(event);
    container.addr = "N/A";

    std::lock_guard<std::mutex> lock(fileMutex_);
    if (handler_->Init()) {
        handler_->Write(&container, CallStackString{event->cCallStack, event->pyCallStack});
    }
}

void Dump::DumpOpLaunchEvent(std::shared_ptr<OpLaunchEvent>& event)
{
    auto container = EventToContainer(event);
    container.addr = "N/A";

    // 组装attr
    std::string attr = event->attr.empty() ? "" : event->attr;
    if (!attr.empty()) {
        container.attr = "\"{" + attr + "}\"";
    }

    std::lock_guard<std::mutex> lock(fileMutex_);
    if (handler_->Init()) {
        handler_->Write(&container, CallStackString{event->cCallStack, event->pyCallStack});
    }
}

void Dump::DumpKernelLaunchEvent(std::shared_ptr<KernelLaunchEvent>& event)
{
    auto container = EventToContainer(event);
    container.addr = "N/A";

    // 组装attr
    std::string attr;
    if (event->eventSubType == EventSubType::ATB_KERNEL_START || event->eventSubType == EventSubType::ATB_KERNEL_END) {
        attr = event->attr;
    } else {
        attr += "streamId:" + event->streamId + ",";
        attr += "taskId:" + event->taskId;
    }
    container.attr = "\"{" + attr + "}\"";

    std::lock_guard<std::mutex> lock(fileMutex_);
    if (handler_->Init()) {
        handler_->Write(&container, CallStackString{event->cCallStack, event->pyCallStack});
    }
}

void Dump::DumpSystemEvent(std::shared_ptr<SystemEvent>& event)
{
    auto container = EventToContainer(event);
    container.addr = "N/A";

    std::lock_guard<std::mutex> lock(fileMutex_);
    if (handler_->Init()) {
        handler_->Write(&container, CallStackString{event->cCallStack, event->pyCallStack});
    }
}
}