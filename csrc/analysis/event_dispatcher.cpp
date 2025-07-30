// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "event_dispatcher.h"

#include <algorithm>
#include <vector>

#include "utility/log.h"

namespace Leaks {

EventDispatcher& EventDispatcher::GetInstance()
{
    static EventDispatcher dispatcher;
    return dispatcher;
}

void EventDispatcher::Subscribe(const SubscriberId& id,
    const std::vector<EventBaseType>& eventTypes, const Priority& priority, const HandlerFunc& func)
{
    Subscriber newSubscriber{id, priority, func};

    for (auto eventType : eventTypes) {
        if (eventSubscribers_.find(eventType) == eventSubscribers_.end()) {
            eventSubscribers_[eventType] = {};
        }
        auto& subscribers = eventSubscribers_[eventType];
        auto subscriberIt = std::find(subscribers.begin(), subscribers.end(), id);
        if (subscriberIt == subscribers.end()) {
            // 按序插入
            auto it = std::lower_bound(subscribers.begin(), subscribers.end(), newSubscriber);
            subscribers.insert(it, newSubscriber);
        }
    }
}

void EventDispatcher::UnSubscribe(const SubscriberId& id)
{
    for (auto& pair : eventSubscribers_) {
        auto& subscribers = pair.second;
        auto subIt = std::find(subscribers.begin(), subscribers.end(), id);
        if (subIt != subscribers.end()) {
            subscribers.erase(subIt);
        }
    }
}

void EventDispatcher::DispatchEvent(std::shared_ptr<EventBase>& event, MemoryState* state)
{
    auto it = eventSubscribers_.find(event->eventType);
    if (it != eventSubscribers_.end()) {
        auto& subscribers = it->second;
        for (const auto& subscriber : subscribers) {
            subscriber.handler(event, state);
        }
    }
}

}