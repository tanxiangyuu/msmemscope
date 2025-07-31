// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef EVENT_DISPATCHER_H
#define EVENT_DISPATCHER_H

#include <unordered_map>
#include <functional>

#include "event.h"
#include "memory_state_manager.h"

namespace Leaks {

enum class SubscriberId : uint8_t {
    DECOMPOSE_ANALYZER = 0,
    INEFFICIENT_ANALYZER,
    LEAKS_ANALYZER,
    DUMP,
};

class EventDispatcher {
public:
    enum class Priority : uint8_t {
        High = 3,           // decompose, inefficient, leaks
        Medium = 2,
        Low = 1,
        Lowest = 0,         // dump
    };

    using HandlerFunc = std::function<void(std::shared_ptr<EventBase>&, MemoryState*)>;

    struct Subscriber {
        SubscriberId id;
        Priority priority;
        HandlerFunc handler;

        // 用于排序，优先级高的排在前面
        bool operator<(const Subscriber& other) const
        {
            return static_cast<uint8_t>(priority) > static_cast<uint8_t>(other.priority);
        }

        // 用于查找
        bool operator==(SubscriberId otherId) const
        {
            return id == otherId;
        }
    };

    static EventDispatcher& GetInstance();
    void DispatchEvent(std::shared_ptr<EventBase>& event, MemoryState* state);
    void Subscribe(const SubscriberId& id,
        const std::vector<EventBaseType>& eventTypes, const Priority& priority, const HandlerFunc& func);
    void UnSubscribe(const SubscriberId& id);
private:
    EventDispatcher() = default;
    ~EventDispatcher() = default;

    EventDispatcher(const EventDispatcher&) = delete;
    EventDispatcher& operator=(const EventDispatcher&) = delete;
    EventDispatcher(EventDispatcher&&) = delete;
    EventDispatcher& operator=(EventDispatcher&&) = delete;

    std::unordered_map<EventBaseType, std::vector<Subscriber>> eventSubscribers_;
};

}

#endif