// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include <gtest/gtest.h>
#define private public
#include "event_trace/atb_op_watch/atb_op_excute_watch.h"
#undef private
#include "event_trace/atb_op_watch/atb_tensor_dump.h"
#include "securec.h"

using namespace Leaks;

constexpr uint64_t DATA_SIZE_1 = 100;
constexpr uint64_t DATA_SIZE_2 = 200;
constexpr uint64_t DATA_SIZE_3 = 1000;
constexpr uint64_t DATA_SIZE_4 = 2000;
constexpr uint64_t DATA_SIZE_5 = 10000;
constexpr uint64_t DATA_SIZE_6 = 20000;
constexpr uint64_t DATA_VALUE_1 = 123456;
constexpr uint64_t DATA_VALUE_2 = 654321;
constexpr uint64_t DATA_VALUE_3 = 1234560;
constexpr uint64_t DATA_VALUE_4 = 6543210;
constexpr uint64_t DATA_VALUE_5 = 12345600;
constexpr uint64_t DATA_VALUE_6 = 65432100;

void ATBOpExcuteWatchNormalCheckOpA()
{
    auto &instance = ATBOpExcuteWatch::GetInstance();
    auto watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    instance.AtbOpExcuteBegin("A");

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    atb::Tensor tesor1 = {};
    tesor1.deviceData = reinterpret_cast<void *>(DATA_VALUE_1);
    tesor1.dataSize = DATA_SIZE_1;

    atb::Tensor tesor2 = {};
    tesor2.deviceData = reinterpret_cast<void *>(DATA_VALUE_2);
    tesor2.dataSize = DATA_SIZE_2;

    atb::SVector<atb::Tensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.AtbOpExcuteEnd("A", tensors);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);
}

void ATBOpExcuteWatchNormalCheckOpB()
{
    auto &instance = ATBOpExcuteWatch::GetInstance();
    auto watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    instance.AtbOpExcuteBegin("first");

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    atb::Tensor tesor1 = {};
    tesor1.deviceData = reinterpret_cast<void *>(DATA_VALUE_1);
    tesor1.dataSize = DATA_SIZE_1;

    atb::Tensor tesor2 = {};
    tesor2.deviceData = reinterpret_cast<void *>(DATA_VALUE_2);
    tesor2.dataSize = DATA_SIZE_2;

    atb::SVector<atb::Tensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.AtbOpExcuteEnd("first", tensors);

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors[0].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[0].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[1].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[1].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);
}

void ATBOpExcuteWatchNormalCheckOpC()
{
    auto &instance = ATBOpExcuteWatch::GetInstance();
    auto watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors[0].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[0].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[1].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[1].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    EXPECT_EQ(instance.IsInMonitoring(), true);

    instance.AtbOpExcuteBegin("C");

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors[0].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[0].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[1].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[1].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);

    atb::Tensor tesor1 = {};
    tesor1.deviceData = reinterpret_cast<void *>(DATA_VALUE_3);
    tesor1.dataSize = DATA_SIZE_3;

    atb::Tensor tesor2 = {};
    tesor2.deviceData = reinterpret_cast<void *>(DATA_VALUE_4);
    tesor2.dataSize = DATA_SIZE_4;

    atb::SVector<atb::Tensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.AtbOpExcuteEnd("C", tensors);

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors[0].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[0].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[1].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[1].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);
}

void ATBOpExcuteWatchNormalCheckOpD()
{
    auto &instance = ATBOpExcuteWatch::GetInstance();
    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    auto watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors[0].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[0].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[1].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[1].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);

    instance.AtbOpExcuteBegin("last");

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors[0].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[0].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[1].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[1].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);

    atb::Tensor tesor1 = {};
    tesor1.deviceData = reinterpret_cast<void *>(DATA_VALUE_5);
    tesor1.dataSize = DATA_SIZE_5;

    atb::Tensor tesor2 = {};
    tesor2.deviceData = reinterpret_cast<void *>(DATA_VALUE_6);
    tesor2.dataSize = DATA_SIZE_6;

    atb::SVector<atb::Tensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.AtbOpExcuteEnd("last", tensors);

    watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);
}

void ATBOpExcuteWatchNormalCheckOpE()
{
    auto &instance = ATBOpExcuteWatch::GetInstance();
    auto watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    instance.AtbOpExcuteBegin("E");

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    atb::Tensor tesor1 = {};
    tesor1.deviceData = reinterpret_cast<void *>(DATA_VALUE_1);
    tesor1.dataSize = DATA_SIZE_1;

    atb::Tensor tesor2 = {};
    tesor2.deviceData = reinterpret_cast<void *>(DATA_VALUE_2);
    tesor2.dataSize = DATA_SIZE_2;

    atb::SVector<atb::Tensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.AtbOpExcuteEnd("E", tensors);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);
}

// 分别执行ABCDE五个OP，其中B和D为start和end算子，校验整个过程中系统的状态是否符合预期
TEST(ATBOpExcuteWatch, ATBOpExcuteWatchNormalCase)
{
    auto &instance = ATBOpExcuteWatch::GetInstance();
    instance.fistWatchOp_ = "first";
    instance.lastWatchOp_ = "last";
    instance.outputId_ = UINT32_MAX;

    ATBOpExcuteWatchNormalCheckOpA();
    ATBOpExcuteWatchNormalCheckOpB();
    ATBOpExcuteWatchNormalCheckOpC();
    ATBOpExcuteWatchNormalCheckOpD();
    ATBOpExcuteWatchNormalCheckOpE();
}

void ATBKernelExcuteWatchNormalCheckOpA()
{
    auto &instance = ATBOpExcuteWatch::GetInstance();
    auto watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    std::string name = "/A/before";
    Mki::Tensor tensor1 = {};
    tensor1.dataSize = static_cast<size_t>(DATA_SIZE_1);
    tensor1.data = reinterpret_cast<void *>(DATA_VALUE_1);

    Mki::Tensor tensor2 = {};
    tensor2.dataSize = static_cast<size_t>(DATA_SIZE_2);
    tensor2.data = reinterpret_cast<void *>(DATA_VALUE_2);

    Mki::SVector<Mki::Tensor> tensors = {};
    tensors.push_back(tensor1);
    tensors.push_back(tensor2);

    instance.AtbKernelExcute(name, tensors);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    name = "/A/after";
    instance.AtbKernelExcute(name, tensors);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);
}

void ATBKernelExcuteWatchNormalCheckOpB()
{
    auto &instance = ATBOpExcuteWatch::GetInstance();
    auto watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    std::string name = "/first/before";
    Mki::Tensor tensor1 = {};
    tensor1.dataSize = static_cast<size_t>(DATA_SIZE_1);
    tensor1.data = reinterpret_cast<void *>(DATA_VALUE_1);

    Mki::Tensor tensor2 = {};
    tensor2.dataSize = static_cast<size_t>(DATA_SIZE_2);
    tensor2.data = reinterpret_cast<void *>(DATA_VALUE_2);

    Mki::SVector<Mki::Tensor> tensors = {};
    tensors.push_back(tensor1);
    tensors.push_back(tensor2);

    instance.AtbKernelExcute(name, tensors);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    name = "/first/after";
    instance.AtbKernelExcute(name, tensors);

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors[0].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[0].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[1].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[1].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);
}

void ATBKernelExcuteWatchNormalCheckOpC()
{
    auto &instance = ATBOpExcuteWatch::GetInstance();
    auto watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors[0].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[0].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[1].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[1].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    EXPECT_EQ(instance.IsInMonitoring(), true);

    std::string name = "/C/before";
    Mki::Tensor tensor1 = {};
    tensor1.dataSize = static_cast<size_t>(DATA_SIZE_1);
    tensor1.data = reinterpret_cast<void *>(DATA_VALUE_1);

    Mki::Tensor tensor2 = {};
    tensor2.dataSize = static_cast<size_t>(DATA_SIZE_2);
    tensor2.data = reinterpret_cast<void *>(DATA_VALUE_2);

    Mki::SVector<Mki::Tensor> tensors = {};
    tensors.push_back(tensor1);
    tensors.push_back(tensor2);

    instance.AtbKernelExcute(name, tensors);

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors[0].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[0].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[1].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[1].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);

    name = "/C/after";
    instance.AtbKernelExcute(name, tensors);

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors[0].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[0].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[1].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[1].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);
}

void ATBKernelExcuteWatchNormalCheckOpD()
{
    auto &instance = ATBOpExcuteWatch::GetInstance();
    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    auto watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors[0].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[0].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[1].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[1].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);

    std::string name = "/last/before";
    Mki::Tensor tensor1 = {};
    tensor1.dataSize = static_cast<size_t>(DATA_SIZE_1);
    tensor1.data = reinterpret_cast<void *>(DATA_VALUE_1);

    Mki::Tensor tensor2 = {};
    tensor2.dataSize = static_cast<size_t>(DATA_SIZE_2);
    tensor2.data = reinterpret_cast<void *>(DATA_VALUE_2);

    Mki::SVector<Mki::Tensor> tensors = {};
    tensors.push_back(tensor1);
    tensors.push_back(tensor2);

    instance.AtbKernelExcute(name, tensors);

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors[0].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[0].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[1].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[1].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);

    name = "/last/after";
    instance.AtbKernelExcute(name, tensors);

    watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);
}

void ATBKernelExcuteWatchNormalCheckOpE()
{
    auto &instance = ATBOpExcuteWatch::GetInstance();
    auto watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    std::string name = "/E/before";
    Mki::Tensor tensor1 = {};
    tensor1.dataSize = static_cast<size_t>(DATA_SIZE_1);
    tensor1.data = reinterpret_cast<void *>(DATA_VALUE_1);

    Mki::Tensor tensor2 = {};
    tensor2.dataSize = static_cast<size_t>(DATA_SIZE_2);
    tensor2.data = reinterpret_cast<void *>(DATA_VALUE_2);

    Mki::SVector<Mki::Tensor> tensors = {};
    tensors.push_back(tensor1);
    tensors.push_back(tensor2);

    instance.AtbKernelExcute(name, tensors);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    name = "/E/after";
    instance.AtbKernelExcute(name, tensors);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);
}

// 分别执行ABCDE五个kernel，其中B和D为start和end算子，校验整个过程中系统的状态是否符合预期
TEST(ATBOpExcuteWatch, ATBKernelExcuteWatchNormalCase)
{
    auto &instance = ATBOpExcuteWatch::GetInstance();
    instance.fistWatchOp_ = "first";
    instance.lastWatchOp_ = "last";
    instance.outputId_ = UINT32_MAX;

    ATBKernelExcuteWatchNormalCheckOpA();
    ATBKernelExcuteWatchNormalCheckOpB();
    ATBKernelExcuteWatchNormalCheckOpC();
    ATBKernelExcuteWatchNormalCheckOpD();
    ATBKernelExcuteWatchNormalCheckOpE();
}

// 分别执行ABCDE五个算子（OP和kernel混合），其中B和D为start和end算子，且D和E为kernel，其余为OP
// 校验整个过程中系统的状态是否符合预期
TEST(ATBOpExcuteWatch, ATBOpKernelMixExcuteWatchNormalCase)
{
    auto &instance = ATBOpExcuteWatch::GetInstance();
    instance.fistWatchOp_ = "first";
    instance.lastWatchOp_ = "last";
    instance.outputId_ = UINT32_MAX;

    ATBOpExcuteWatchNormalCheckOpA();
    ATBOpExcuteWatchNormalCheckOpB();
    ATBKernelExcuteWatchNormalCheckOpC();
    ATBKernelExcuteWatchNormalCheckOpD();
    ATBOpExcuteWatchNormalCheckOpE();
}

void ATBOpExcuteWatchNormalCheckOpBWithOutputId()
{
    auto &instance = ATBOpExcuteWatch::GetInstance();
    auto watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    instance.AtbOpExcuteBegin("first");

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    atb::Tensor tesor1 = {};
    tesor1.deviceData = reinterpret_cast<void *>(DATA_VALUE_1);
    tesor1.dataSize = DATA_SIZE_1;

    atb::Tensor tesor2 = {};
    tesor2.deviceData = reinterpret_cast<void *>(DATA_VALUE_2);
    tesor2.dataSize = DATA_SIZE_2;

    atb::SVector<atb::Tensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.AtbOpExcuteEnd("first", tensors);

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors.size(), 1);
    EXPECT_EQ(watchedTensors[0].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[0].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);
}

void ATBOpExcuteWatchNormalCheckOpCWithOutputId()
{
    auto &instance = ATBOpExcuteWatch::GetInstance();
    auto watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors.size(), 1);
    EXPECT_EQ(watchedTensors[0].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[0].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    EXPECT_EQ(instance.IsInMonitoring(), true);

    instance.AtbOpExcuteBegin("C");

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors.size(), 1);
    EXPECT_EQ(watchedTensors[0].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[0].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);

    atb::Tensor tesor1 = {};
    tesor1.deviceData = reinterpret_cast<void *>(DATA_VALUE_3);
    tesor1.dataSize = DATA_SIZE_3;

    atb::Tensor tesor2 = {};
    tesor2.deviceData = reinterpret_cast<void *>(DATA_VALUE_4);
    tesor2.dataSize = DATA_SIZE_4;

    atb::SVector<atb::Tensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.AtbOpExcuteEnd("C", tensors);

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors.size(), 1);
    EXPECT_EQ(watchedTensors[0].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[0].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);
}

void ATBOpExcuteWatchNormalCheckOpDWithOutputId()
{
    auto &instance = ATBOpExcuteWatch::GetInstance();
    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    auto watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors.size(), 1);
    EXPECT_EQ(watchedTensors[0].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[0].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);

    instance.AtbOpExcuteBegin("last");

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors.size(), 1);
    EXPECT_EQ(watchedTensors[0].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[0].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);

    atb::Tensor tesor1 = {};
    tesor1.deviceData = reinterpret_cast<void *>(DATA_VALUE_5);
    tesor1.dataSize = DATA_SIZE_5;

    atb::Tensor tesor2 = {};
    tesor2.deviceData = reinterpret_cast<void *>(DATA_VALUE_6);
    tesor2.dataSize = DATA_SIZE_6;

    atb::SVector<atb::Tensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.AtbOpExcuteEnd("last", tensors);

    watchedTensors = instance.GetWatchedTensors();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);
}

// 分别执行ABCDE五个OP，其中B和D为start和end算子，校验整个过程中系统的状态是否符合预期（设置outputId）
TEST(ATBOpExcuteWatch, ATBOpExcuteWatchSetOutputIdCase)
{
    auto &instance = ATBOpExcuteWatch::GetInstance();
    instance.fistWatchOp_ = "first";
    instance.lastWatchOp_ = "last";
    instance.outputId_ = 1; // 只取下标为1的tensor

    ATBOpExcuteWatchNormalCheckOpA();
    ATBOpExcuteWatchNormalCheckOpBWithOutputId();
    ATBOpExcuteWatchNormalCheckOpCWithOutputId();
    ATBOpExcuteWatchNormalCheckOpDWithOutputId();
    ATBOpExcuteWatchNormalCheckOpE();
}

TEST(ATBOpExcuteWatch, ATBKernelExcuteWatchCheckAtbKernelExcuteFuc)
{
    auto &instance = ATBOpExcuteWatch::GetInstance();
    instance.fistWatchOp_ = "first";
    instance.lastWatchOp_ = "last";
    instance.outputId_ = UINT32_MAX;

    std::string name = "/add/after1";
    Mki::Tensor tensor = {};
    tensor.dataSize = 123;

    Mki::SVector<Mki::Tensor> tensors = {};
    tensors.push_back(tensor);
    instance.AtbKernelExcute(name, tensors);

    std::string firstName = "/first/after";
    instance.AtbKernelExcute(firstName, tensors);

    tensors.push_back(tensor);
    instance.outputId_ = 0;
    instance.AtbKernelExcute(firstName, tensors);
}

TEST(ATBOpExcuteWatch, ATBKernelExcuteWatchCheckAtbOpExcuteBeginFuc)
{
    auto &instance = ATBOpExcuteWatch::GetInstance();
    instance.fistWatchOp_ = "first";
    instance.lastWatchOp_ = "last";
    instance.outputId_ = 0;

    std::string name = "first";

    atb::Tensor tesor1 = {};
    tesor1.deviceData = reinterpret_cast<void *>(DATA_VALUE_5);
    tesor1.dataSize = DATA_SIZE_5;

    atb::Tensor tesor2 = {};
    tesor2.deviceData = reinterpret_cast<void *>(DATA_VALUE_6);
    tesor2.dataSize = DATA_SIZE_6;

    atb::SVector<atb::Tensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.AtbOpExcuteEnd(name, tensors);
}

TEST(ATBOpExcuteWatch, ATBWatchCheckCleanFileNameFunc)
{
    std::string fileName = "/123/123/";
    CleanFileName(fileName);
    EXPECT_EQ(fileName, ".123.123.");
}