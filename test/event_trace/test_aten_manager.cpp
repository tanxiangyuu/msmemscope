// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#define private public
#include "event_trace/event_report.h"
#undef private
#include "bit_field.h"
#include "event_trace/mstx_hooks/mstx_manager.h"
#include "event_trace/mstx_hooks/aten_manager.h"

using namespace Leaks;

TEST(AtenManagerTest, ReportAtenStartLaunchTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    BitField<decltype(instance.config_.eventType)> eventBit;
    BitField<decltype(instance.config_.levelType)> levelBit;
    levelBit.setBit(static_cast<size_t>(LevelType::LEVEL_OP));
    levelBit.setBit(static_cast<size_t>(LevelType::LEVEL_KERNEL));
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::LAUNCH_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::ACCESS_EVENT));
    instance.config_.eventType = eventBit.getValue();
    instance.config_.levelType = levelBit.getValue();
    instance.config_.enableCStack = true;
    instance.config_.enablePyStack = true;
    const char* msg = "leaks-aten-b: {func.__module__}.{func.__name__}";
    uint32_t streamId = 0;
    MstxManager::GetInstance().ReportMarkA(msg, streamId);
}

TEST(AtenManagerTest, ReportAtenStart_nullname_Test) {
    const char* msg = "leaks-aten-b:";
    uint32_t streamId = 0;
    MstxManager::GetInstance().ReportMarkA(msg, streamId);
}

TEST(AtenManagerTest, ReportAtenEndLaunchTest) {
    const char* msg = "leaks-aten-e: {func.__module__}.{func.__name__}";
    uint32_t streamId = 0;
    MstxManager::GetInstance().ReportMarkA(msg, streamId);
}

TEST(AtenManagerTest, ReportAtenUnknownAccessTest) {
    const char* msg = "leaks-ac:ptr=1000;is_write=False;is_read=False;is_output=False;"\
        "name={func.__module__}.{func.__name__};shape={value.shape};dtype={value.dtype};tensor_size=500;device=0";
    uint32_t streamId = 0;
    MstxManager::GetInstance().ReportMarkA(msg, streamId);
}

TEST(AtenManagerTest, ReportAtenWriteAccessTest) {
    const char* msg = "leaks-ac:ptr=1000;is_write=True;is_read=False;is_output=False;"\
        "name={func.__module__}.{func.__name__};shape={value.shape};dtype={value.dtype};tensor_size=500;device=0";
    uint32_t streamId = 0;
    MstxManager::GetInstance().ReportMarkA(msg, streamId);
}

TEST(AtenManagerTest, ReportAtenReadAccessTest) {
    const char* msg = "leaks-ac:ptr=1000;is_write=False;is_read=True;is_output=False;"\
        "name={func.__module__}.{func.__name__};shape={value.shape};dtype={value.dtype};tensor_size=500;device=0";
    uint32_t streamId = 0;
    MstxManager::GetInstance().ReportMarkA(msg, streamId);
}

TEST(AtenManagerTest, ReportAtenReadAccess_nullname_Test) {
    const char* msg = "leaks-ac:ptr=1000;is_write=False;is_read=True;is_output=False;"\
        "name=;shape={value.shape};dtype={value.dtype};tensor_size=500;device=0";
    uint32_t streamId = 0;
    MstxManager::GetInstance().ReportMarkA(msg, streamId);
}

TEST(AtenManagerTest, ExtractTensorInfoTest) {
    const char* msg = "leaks-ac:ptr=1000";
    uint32_t streamId = 0;
    MstxManager::GetInstance().ReportMarkA(msg, streamId);
}

