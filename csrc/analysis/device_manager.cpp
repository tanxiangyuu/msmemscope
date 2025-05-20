// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "device_manager.h"

namespace Leaks {
DeviceManager& DeviceManager::GetInstance(Config config)
{
    static DeviceManager instance(config);
    return instance;
}

DeviceManager::DeviceManager(Config config)
{
    config_ = config;
}

std::map<int32_t, std::shared_ptr<MemoryStateRecord>>& DeviceManager::GetMemoryStateRecordMap()
{
    return memoryStateRecordMap_;
}

std::shared_ptr<MemoryStateRecord> DeviceManager::GetMemoryStateRecord(int32_t deviceId)
{
    auto it = memoryStateRecordMap_.find(deviceId);
    if (it == memoryStateRecordMap_.end()) {
        memoryStateRecordMap_.insert({deviceId, std::make_shared<MemoryStateRecord>(config_)});
    }
    return memoryStateRecordMap_[deviceId];
}
}
