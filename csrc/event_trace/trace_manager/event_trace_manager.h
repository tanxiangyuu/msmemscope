// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
#ifndef EVENT_TRACE_MANAGER_H
#define EVENT_TRACE_MANAGER_H

#include <mutex>
#include <string>
#include <unordered_map>
#include <functional>
#include <atomic>
#include "config_info.h"
#include "record_info.h"

namespace Leaks {

class ConfigManager {
public:
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;

    static ConfigManager& Instance()
    {
        static ConfigManager instance;
        return instance;
    }

    Config GetConfig();
    void InitConfig();
    void InitStartConfig();
    bool SetConfig(const std::unordered_map<std::string, std::string> &config);
    void SetConfig(const Config &config);

private:
    ConfigManager();

    ~ConfigManager() = default;

    void SetConfigImpl(const Config &config);
    void GetConfigAfterInit(Config &config);

    std::mutex mutex_;
    Config config_;
    bool firstConfig = true;
};

inline Config GetConfig()
{
    return ConfigManager::Instance().GetConfig();
}

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
    
    bool IsNeedTrace(const RecordType type = RecordType::INVALID_RECORD);
    bool ShouldTraceType(const RecordType type = RecordType::INVALID_RECORD);
    bool IsTracingEnabled();
    void SetTraceStatus(const EventTraceStatus status); // 通过python接口在运行时动态修改
    void InitJudgeFuncTable();
    void SetAclInitStatus(bool isInit);
    void HandleWithATenCollect();
private:
    EventTraceManager()
    {
        InitTraceStatus();
        InitJudgeFuncTable();
    }
    ~EventTraceManager()
    {
        destroyed_.store(true);
    }

    void InitTraceStatus(); // 命令行拉起时有一个初始化状态

    std::mutex mutex_;
    EventTraceStatus status_ = EventTraceStatus::NOT_IN_TRACING;

    std::atomic<bool> aclInit_{ false };
    std::unordered_map<RecordType, std::function<bool()>> judgeFuncTable_;
    std::atomic<bool> destroyed_{false};
};

}

#endif
