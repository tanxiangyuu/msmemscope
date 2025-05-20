// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#define private public
#include "device_manager.h"
#undef private

#include "memory_state_record.h"

using namespace Leaks;

TEST(DeviceManagerTest, get_empty_memory_state_record_expect_true_data)
{
    int32_t deviceId = 0;
    Config config;
    auto memoryStateRecord = DeviceManager::GetInstance(config).GetMemoryStateRecord(deviceId);
    ASSERT_EQ(DeviceManager::GetInstance(config).memoryStateRecordMap_.size(), 1);
    DeviceManager::GetInstance(config).memoryStateRecordMap_.erase(0);
}

TEST(DeviceManagerTest, get_exist_memory_state_record_expect_true_data)
{
    int32_t deviceId = 1;
    Config config;
    auto record = std::make_shared<MemoryStateRecord>(config);
    DeviceManager::GetInstance(config).memoryStateRecordMap_.insert({1, record});
    auto memoryStateRecord = DeviceManager::GetInstance(config).GetMemoryStateRecord(deviceId);
    ASSERT_EQ(memoryStateRecord, record);
    DeviceManager::GetInstance(config).memoryStateRecordMap_.erase(1);
}
