// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
#ifndef EVENT_TRACE_MANAGER_H
#define EVENT_TRACE_MANAGER_H

#include <mutex>

namespace Leaks {

enum class EventTraceStatus : uint8_t {
    IN_TRACING = 0,
    NOT_IN_TRACING,
};

class EventTraceManager {
public:
    EventTraceManager(const EventTraceManager&) = delete;
    EventTraceManager& operator=(const EventTraceManager&) = delete;

    static EventTraceManager& Instance()
    {
        static EventTraceManager instance;
        return instance;
    }
    
    bool IsNeedTrace();
    void SetTraceStatus(const EventTraceStatus status); // 通过python接口在运行时动态修改
private:
    EventTraceManager()
    {
        InitTraceStatus();
    }
    ~EventTraceManager() = default;

    void InitTraceStatus(); // 命令行拉起时有一个初始化状态
    void HandleWithTraceStatusChanged(const EventTraceStatus status);

    std::mutex mutex_;
    EventTraceStatus status_ = EventTraceStatus::IN_TRACING;
};

}

#endif
