// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <string>
#include <unordered_map>
#include <memory>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <iostream>
#define private public
#include "dump.h"
#include "decompose_analyzer.h"
#include "data_handler.h"
#include "process.h"
#undef private
#include "record_info.h"
#include "config_info.h"
#include "securec.h"
#include "file.h"
#include "event.h"
#include "event_dispatcher.h"
#include "memory_state_manager.h"
 
using namespace Leaks;
 
std::unordered_map<std::string, std::shared_ptr<EventBase>> eventMap;
 
static void ResetSingleton()
{
    // 全局变量初始化
    Config config;
    DecomposeAnalyzer::GetInstance();
    Dump::GetInstance(config);
 
    // 取消数据订阅与部分config参数
    EventDispatcher::GetInstance().UnSubscribe(SubscriberId::DECOMPOSE_ANALYZER);
}
 
// 定义测试夹具
class TestProcess : public ::testing::Test {
protected:
    void SetUp() override
    {
        // 初始化单例类的参数
        ResetSingleton();
 
        AddHostMemoryEvent();
        AddHalHostMemoryEvent();
        AddHalDeviceMemoryEvent();
 
        AddAccessEvent();
 
        AddPtaMemoryEvent();
        AddAtbMemoryEvent();
        AddMindsporeMemoryEvent();
    }
 
    void TearDown() override
    {
        std::unordered_map<std::string, std::shared_ptr<EventBase>>().swap(eventMap);
    }
 
private:
    void AddHostMemoryEvent()
    {
        auto event0 = std::make_shared<MemoryEvent>();
        event0->poolType = PoolType::HOST;
        event0->id = 0;
        event0->timestamp = 0;
        event0->pid = 123;
        event0->tid = 1234;
        event0->addr = 123456;
        event0->device = "host";
        event0->name = "N/A";
        event0->eventType = EventBaseType::MALLOC;
        event0->eventSubType = EventSubType::HOST;
        event0->size = 10;
        eventMap["HostUnknownMallocEvent"] = event0;
 
        auto event1 = std::make_shared<MemoryEvent>();
        event1->poolType = PoolType::HOST;
        event1->id = 3;
        event1->timestamp = 3;
        event1->pid = 123;
        event1->tid = 1234;
        event1->addr = 123456;
        event1->device = "host";
        event1->name = "N/A";
        event1->eventType = EventBaseType::MALLOC;
        event1->eventSubType = EventSubType::HOST;
        event1->size = 10;
        eventMap["HostMallocEvent"] = event1;
 
        auto event2 = std::make_shared<MemoryEvent>();
        event2->poolType = PoolType::HOST;
        event2->id = 54;
        event2->timestamp = 54;
        event2->pid = 123;
        event2->tid = 1234;
        event2->addr = 123456;
        event2->device = "host";
        event2->name = "N/A";
        event2->eventType = EventBaseType::FREE;
        event2->eventSubType = EventSubType::HOST;
        eventMap["HostFreeEvent"] = event2;
    }
 
    void AddHalHostMemoryEvent()
    {
        auto event1 = std::make_shared<MemoryEvent>();
        event1->poolType = PoolType::HAL;
        event1->id = 3;
        event1->timestamp = 3;
        event1->pid = 123;
        event1->tid = 1234;
        event1->addr = 12345;
        event1->device = "host";
        event1->name = "N/A";
        event1->eventType = EventBaseType::MALLOC;
        event1->eventSubType = EventSubType::HAL;
        event1->size = 10;
        event1->moduleId = 100;
        eventMap["HalHostMallocEvent"] = event1;
 
        auto event2 = std::make_shared<MemoryEvent>();
        event2->poolType = PoolType::HAL;
        event2->id = 54;
        event2->timestamp = 54;
        event2->pid = 123;
        event2->tid = 1234;
        event2->addr = 12345;
        event2->device = "N/A";
        event2->name = "N/A";
        event2->eventType = EventBaseType::FREE;
        event2->eventSubType = EventSubType::HAL;
        event2->moduleId = 100;
        eventMap["HalHostFreeEvent"] = event2;
    }
 
    void AddHalDeviceMemoryEvent()
    {
        auto event1 = std::make_shared<MemoryEvent>();
        event1->poolType = PoolType::HAL;
        event1->id = 3;
        event1->timestamp = 3;
        event1->pid = 123;
        event1->tid = 1234;
        event1->addr = 12345;
        event1->device = "0";
        event1->name = "N/A";
        event1->eventType = EventBaseType::MALLOC;
        event1->eventSubType = EventSubType::HAL;
        event1->size = 10;
        event1->moduleId = 1;
        eventMap["HalDeviceMallocEvent"] = event1;
 
        auto event2 = std::make_shared<MemoryEvent>();
        event2->poolType = PoolType::HAL;
        event2->id = 54;
        event2->timestamp = 54;
        event2->pid = 123;
        event2->tid = 1234;
        event2->addr = 12345;
        event2->device = "N/A";
        event2->name = "N/A";
        event2->eventType = EventBaseType::FREE;
        event2->eventSubType = EventSubType::HAL;
        event2->moduleId = 1;
        eventMap["HalDeviceFreeEvent"] = event2;
    }
 
    void AddAccessEvent()
    {
        auto event1 = std::make_shared<MemoryEvent>();
        event1->poolType = PoolType::PTA;
        event1->id = 13;
        event1->timestamp = 13;
        event1->pid = 123;
        event1->tid = 1234;
        event1->addr = 12345;
        event1->device = "0";
        event1->name = "aten.add";
        event1->eventType = EventBaseType::ACCESS;
        event1->eventSubType = EventSubType::ATEN_READ_OR_WRITE;
        event1->size = 10;
        event1->attr = "dtype:torch.float16,shape:torch.Size([1,5])";
        eventMap["PtaAccessEvent"] = event1;
 
        auto event2 = std::make_shared<MemoryEvent>();
        event2->poolType = PoolType::ATB;
        event2->id = 14;
        event2->timestamp = 14;
        event2->pid = 123;
        event2->tid = 1234;
        event2->addr = 12345;
        event2->device = "0";
        event2->name = "addOperation";
        event2->eventType = EventBaseType::ACCESS;
        event2->eventSubType = EventSubType::ATB_READ_OR_WRITE;
        event2->size = 10;
        event2->attr = "dtype:FLOAT,format:NZD,shape:1 5 ,type:atb tensor";
        eventMap["AtbAccessEvent"] = event2;
    }
 
    void AddPtaMemoryEvent()
    {
        auto event1 = std::make_shared<MemoryEvent>();
        event1->poolType = PoolType::PTA;
        event1->id = 3;
        event1->timestamp = 3;
        event1->pid = 123;
        event1->tid = 1234;
        event1->addr = 12345;
        event1->device = "0";
        event1->name = "N/A";
        event1->eventType = EventBaseType::MALLOC;
        event1->eventSubType = EventSubType::PTA;
        event1->size = 10;
        event1->total = 10;
        event1->used = 10;
        event1->cCallStack = "func1\nfunc2";
        event1->pyCallStack = "func3\nfunc4";
        eventMap["PtaMallocEvent"] = event1;
 
        auto event2 = std::make_shared<MemoryEvent>();
        event2->poolType = PoolType::PTA;
        event2->id = 54;
        event2->timestamp = 54;
        event2->pid = 123;
        event2->tid = 1234;
        event2->addr = 12345;
        event2->device = "0";
        event2->name = "N/A";
        event2->eventType = EventBaseType::FREE;
        event2->eventSubType = EventSubType::PTA;
        event2->size = 10;
        event2->total = 0;
        event2->used = 0;
        eventMap["PtaFreeEvent"] = event2;
    }
 
    void AddAtbMemoryEvent()
    {
        auto event1 = std::make_shared<MemoryEvent>();
        event1->poolType = PoolType::ATB;
        event1->id = 3;
        event1->timestamp = 3;
        event1->pid = 123;
        event1->tid = 1234;
        event1->addr = 12345;
        event1->device = "0";
        event1->name = "N/A";
        event1->eventType = EventBaseType::MALLOC;
        event1->eventSubType = EventSubType::ATB;
        event1->size = 10;
        event1->total = 10;
        event1->used = 10;
        event1->cCallStack = "func1\nfunc2";
        event1->pyCallStack = "func3\nfunc4";
        eventMap["AtbMallocEvent"] = event1;
 
        auto event2 = std::make_shared<MemoryEvent>();
        event2->poolType = PoolType::ATB;
        event2->id = 54;
        event2->timestamp = 54;
        event2->pid = 123;
        event2->tid = 1234;
        event2->addr = 12345;
        event2->device = "0";
        event2->name = "N/A";
        event2->eventType = EventBaseType::FREE;
        event2->eventSubType = EventSubType::ATB;
        event2->size = 10;
        event2->total = 0;
        event2->used = 0;
        eventMap["AtbFreeEvent"] = event2;
    }
 
    void AddMindsporeMemoryEvent()
    {
        auto event1 = std::make_shared<MemoryEvent>();
        event1->poolType = PoolType::MINDSPORE;
        event1->id = 3;
        event1->timestamp = 3;
        event1->pid = 123;
        event1->tid = 1234;
        event1->addr = 12345;
        event1->device = "0";
        event1->name = "N/A";
        event1->eventType = EventBaseType::MALLOC;
        event1->eventSubType = EventSubType::MINDSPORE;
        event1->size = 10;
        event1->total = 10;
        event1->used = 10;
        event1->cCallStack = "func1\nfunc2";
        event1->pyCallStack = "func3\nfunc4";
        eventMap["MindsporeMallocEvent"] = event1;
 
        auto event2 = std::make_shared<MemoryEvent>();
        event2->poolType = PoolType::MINDSPORE;
        event2->id = 54;
        event2->timestamp = 54;
        event2->pid = 123;
        event2->tid = 1234;
        event2->addr = 12345;
        event2->device = "0";
        event2->name = "N/A";
        event2->eventType = EventBaseType::FREE;
        event2->eventSubType = EventSubType::MINDSPORE;
        event2->size = 10;
        event2->total = 0;
        event2->used = 0;
        eventMap["MindsporeFreeEvent"] = event2;
    }
};