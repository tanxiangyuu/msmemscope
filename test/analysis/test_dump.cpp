// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include <gtest/gtest.h>
#include "dump.h"
#include <string>
#include "event_dispatcher.h"

namespace MemScope {
class DumpTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        config_ = Config();
        dumper = &Dump::GetInstance(config_);
    }
    Config config_;
    Dump* dumper;
};

TEST_F(DumpTest, GetInstance_Singleton)
{
    // 判断不同的config是否返回同一个dump（只会初始化一次）,后续调用直接返回该实例的引用
    Dump* instance1 = &Dump::GetInstance(config_);
    Dump* instance2 = &Dump::GetInstance(config_);
    Config configTmp = Config();
    configTmp.enableCompare = true;
    Dump* instance3 = &Dump::GetInstance(configTmp);
    EXPECT_EQ(instance1, instance2);
    EXPECT_EQ(instance1, instance3);
}

TEST_F(DumpTest, EventHandle_Interface)
{
    auto memoryEvent = std::make_shared<MemoryEvent>();
    memoryEvent->eventType = EventBaseType::MALLOC;
    memoryEvent->eventSubType = EventSubType::HAL;
    memoryEvent->describeOwner = "hal";

    std::shared_ptr<EventBase> eventBase = memoryEvent;
    MemoryState* state = new MemoryState();

    dumper->EventHandle(eventBase, state);
}

TEST_F(DumpTest, WritePublicEventToFile_Interface)
{
    auto memoryEvent = std::make_shared<MemoryEvent>();
    memoryEvent->eventType = EventBaseType::MALLOC;
    memoryEvent->eventSubType = EventSubType::HAL;
    memoryEvent->describeOwner = "hal";

    std::shared_ptr<EventBase> eventBase = memoryEvent;
    MemoryState* state = new MemoryState();

    // 测试接口
    dumper->EventHandle(eventBase, state);
    dumper->WritePublicEventToFile();
}

TEST_F(DumpTest, FflushEventToFile_Interface)
{
    auto memoryEvent = std::make_shared<MemoryEvent>();
    memoryEvent->eventType = EventBaseType::MALLOC;
    memoryEvent->eventSubType = EventSubType::HAL;
    memoryEvent->describeOwner = "hal";

    std::shared_ptr<EventBase> eventBase = memoryEvent;
    MemoryState* state = new MemoryState();

    // 测试接口
    dumper->EventHandle(eventBase, state);
    dumper->FflushEventToFile();
}

}