// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef DEVICE_MANGER_H
#define DEVICE_MANGER_H

#include <map>
#include <memory>

#include "memory_state_record.h"

namespace Leaks {
class DeviceManager {
public:
    static DeviceManager& GetInstance(Config config);
    std::shared_ptr<MemoryStateRecord>& GetMemoryStateRecord(int32_t deviceId);
    std::map<int32_t, std::shared_ptr<MemoryStateRecord>>& GetMemoryStateRecordMap();
    explicit DeviceManager(Config config);
private:
    ~DeviceManager() = default;
private:
    std::map<int32_t, std::shared_ptr<MemoryStateRecord>> memoryStateRecordMap_;
    Config config_;
};

}
#endif
