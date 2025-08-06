// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include <gtest/gtest.h>
#include "inefficient_analyzer.h"
#include <string>
#include "event_dispatcher.h"

// 假设以下类和结构体已定义
namespace Leaks {
// 测试用例
class InefficientAnalyzerTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        analyzer = &InefficientAnalyzer::GetInstance();
    }

    InefficientAnalyzer* analyzer;
};

TEST_F(InefficientAnalyzerTest, TestOPLaunchStartEnd)
{
    uint64_t pid = 1234;

    // ATB_START
    auto eventStart = std::make_shared<EventBase>();
    eventStart->eventType = EventBaseType::OP_LAUNCH;
    eventStart->eventSubType = EventSubType::ATB_START;
    eventStart->pid = pid;
    MemoryState state;
    analyzer->EventHandle(eventStart, &state);

    // ATB_END
    auto eventEnd = std::make_shared<EventBase>();
    eventEnd->eventType = EventBaseType::OP_LAUNCH;
    eventEnd->eventSubType = EventSubType::ATB_END;
    eventEnd->pid = pid;
    analyzer->EventHandle(eventEnd, &state);
}

TEST_F(InefficientAnalyzerTest, TestMALLOCEventHandling)
{
    uint64_t pid = 1234;

    auto event = std::make_shared<EventBase>();
    event->eventType = EventBaseType::MALLOC;
    event->eventSubType = EventSubType::ATB;
    event->pid = pid;
    MemoryState state;
    analyzer->EventHandle(event, &state);
}

TEST_F(InefficientAnalyzerTest, TestACCESSEventTemporaryIdleness)
{
    uint64_t pid = 1234;

    auto mallocEvent = std::make_shared<EventBase>();
    mallocEvent->eventType = EventBaseType::MALLOC;
    mallocEvent->eventSubType = EventSubType::ATB;
    mallocEvent->pid = pid;
    auto event1 = std::make_shared<EventBase>();
    event1->eventType = EventBaseType::ACCESS;
    event1->eventSubType = EventSubType::ATB;
    event1->pid = pid;
    auto event2 = std::make_shared<EventBase>();
    event2->eventType = EventBaseType::ACCESS;
    event2->eventSubType = EventSubType::ATB;
    event2->pid = pid;
    MemoryState state;
    auto memMallocEvent = std::dynamic_pointer_cast<MemoryEvent>(mallocEvent);
    auto memEvent1 = std::dynamic_pointer_cast<MemoryEvent>(event1);
    auto memEvent2 = std::dynamic_pointer_cast<MemoryEvent>(event2);
    state.events.push_back(memMallocEvent);
    state.events.push_back(memEvent1);
    state.events.push_back(memEvent2);
    state.apiId.push_back(0);
    state.apiId.push_back(1);
    state.apiId.push_back(200);

    analyzer->EventHandle(event2, &state);
}
} // namespace