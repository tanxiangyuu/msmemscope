// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include <gtest/gtest.h>

#define private public
#include "dump.h"
#undef private

#include <string>
#include "bit_field.h"
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

// ==================== RFC: Precise Event Filtering Tests ====================

TEST_F(DumpTest, ShouldDumpEvent_only_launch_enabled)
{
    // 设置 dumpEventType = LAUNCH_EVENT (bit 2)，直接修改单例内部的 config_
    BitField<decltype(dumper->config_.dumpEventType)> dumpBit;
    dumpBit.setBit(static_cast<size_t>(EventType::LAUNCH_EVENT));
    dumper->config_.dumpEventType = dumpBit.getValue();

    // MALLOC / FREE 不应落盘
    EXPECT_FALSE(dumper->ShouldDumpEvent(EventBaseType::MALLOC));
    EXPECT_FALSE(dumper->ShouldDumpEvent(EventBaseType::FREE));
    EXPECT_FALSE(dumper->ShouldDumpEvent(EventBaseType::ACCESS));

    // OP_LAUNCH / KERNEL_LAUNCH 应落盘（映射到 LAUNCH_EVENT）
    EXPECT_TRUE(dumper->ShouldDumpEvent(EventBaseType::OP_LAUNCH));
    EXPECT_TRUE(dumper->ShouldDumpEvent(EventBaseType::KERNEL_LAUNCH));

    // 不可控事件始终落盘
    EXPECT_TRUE(dumper->ShouldDumpEvent(EventBaseType::MSTX));
    EXPECT_TRUE(dumper->ShouldDumpEvent(EventBaseType::SYSTEM));
    EXPECT_TRUE(dumper->ShouldDumpEvent(EventBaseType::SNAPSHOT));
    EXPECT_TRUE(dumper->ShouldDumpEvent(EventBaseType::CLEAN_UP));
    EXPECT_TRUE(dumper->ShouldDumpEvent(EventBaseType::MEMORY_OWNER));
}

TEST_F(DumpTest, ShouldDumpEvent_all_disabled)
{
    // dumpEventType = 0，不落盘任何用户可控事件，直接修改单例内部的 config_
    dumper->config_.dumpEventType = 0;

    EXPECT_FALSE(dumper->ShouldDumpEvent(EventBaseType::MALLOC));
    EXPECT_FALSE(dumper->ShouldDumpEvent(EventBaseType::FREE));
    EXPECT_FALSE(dumper->ShouldDumpEvent(EventBaseType::OP_LAUNCH));
    EXPECT_FALSE(dumper->ShouldDumpEvent(EventBaseType::KERNEL_LAUNCH));
    EXPECT_FALSE(dumper->ShouldDumpEvent(EventBaseType::ACCESS));

    // 不可控事件仍然落盘
    EXPECT_TRUE(dumper->ShouldDumpEvent(EventBaseType::MSTX));
    EXPECT_TRUE(dumper->ShouldDumpEvent(EventBaseType::SYSTEM));
}

TEST_F(DumpTest, ShouldDumpEvent_all_enabled)
{
    // dumpEventType = 所有用户可控事件，直接修改单例内部的 config_
    BitField<decltype(dumper->config_.dumpEventType)> dumpBit;
    dumpBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    dumpBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    dumpBit.setBit(static_cast<size_t>(EventType::LAUNCH_EVENT));
    dumpBit.setBit(static_cast<size_t>(EventType::ACCESS_EVENT));
    dumper->config_.dumpEventType = dumpBit.getValue();

    EXPECT_TRUE(dumper->ShouldDumpEvent(EventBaseType::MALLOC));
    EXPECT_TRUE(dumper->ShouldDumpEvent(EventBaseType::FREE));
    EXPECT_TRUE(dumper->ShouldDumpEvent(EventBaseType::OP_LAUNCH));
    EXPECT_TRUE(dumper->ShouldDumpEvent(EventBaseType::KERNEL_LAUNCH));
    EXPECT_TRUE(dumper->ShouldDumpEvent(EventBaseType::ACCESS));
    EXPECT_TRUE(dumper->ShouldDumpEvent(EventBaseType::MSTX));
}

}