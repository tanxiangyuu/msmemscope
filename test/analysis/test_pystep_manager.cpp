// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <string>
#include <memory>
#include "py_step_manager.h"
#include "record_info.h"
#include "config_info.h"

using namespace MemScope;

// 预期能够正常接收来自python接口的step信息
TEST(PyStepManagerTest, do_msleaks_step_expect_success) {
    ClientId clientId = 0;
    auto firstPyStepRecord = PyStepRecord {};
    firstPyStepRecord.stepId = 1;

    auto SecondPyStepRecord = PyStepRecord {};
    SecondPyStepRecord.stepId = 2;

    auto ThirdPyStepRecord = PyStepRecord {};
    ThirdPyStepRecord.stepId = 3;

    // 分别模拟三次step信息进入
    PyStepManager::Instance().RecordPyStep(clientId, firstPyStepRecord);
    PyStepManager::Instance().RecordPyStep(clientId, SecondPyStepRecord);
    PyStepManager::Instance().RecordPyStep(clientId, ThirdPyStepRecord);
}

// 预期能够正常注册和注销观察者模块
TEST(PyStepManagerTest, do_analyzer_register_and_unregister_expect_success) {
    PyStepManager::Instance().Subscribe(PyStepEventSubscriber::STEP_INNER_ANALYZER, nullptr);
    PyStepManager::Instance().UnSubscribe(PyStepEventSubscriber::STEP_INNER_ANALYZER);
}

// 预期能够正常分发信息给各位观察者
TEST(PyStepManagerTest, do_analyzer_notify_expect_success) {
    PyStepManager::Instance().Subscribe(PyStepEventSubscriber::STEP_INNER_ANALYZER, nullptr);

    ClientId clientId = 0;
    auto firstPyStepRecord = PyStepRecord {};
    firstPyStepRecord.stepId = 1;

    PyStepManager::Instance().RecordPyStep(clientId, firstPyStepRecord);
    PyStepManager::Instance().UnSubscribe(PyStepEventSubscriber::STEP_INNER_ANALYZER);
}
