// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include <gtest/gtest.h>
#define private public
#include "event_trace/op_watch/op_excute_watch.h"
#undef private
#include "event_trace/op_watch/tensor_dumper.h"
#include "event_trace/op_watch/tensor_monitor.h"
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
aclrtStream stream = nullptr;

void OpExcuteWatchNormalCheckOpA()
{
    auto &instance = OpExcuteWatch::GetInstance();
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    instance.OpExcuteBegin(stream, "A", OpType::ATB);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    instance.OpExcuteBegin(stream, "A", OpType::ATEN);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    MonitoredTensor tesor1 = {};
    tesor1.data = reinterpret_cast<void *>(DATA_VALUE_1);
    tesor1.dataSize = DATA_SIZE_1;

    MonitoredTensor tesor2 = {};
    tesor2.data = reinterpret_cast<void *>(DATA_VALUE_2);
    tesor2.dataSize = DATA_SIZE_2;

    std::vector<MonitoredTensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.OpExcuteEnd(stream, "A", tensors, OpType::ATB);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    instance.OpExcuteEnd(stream, "A", tensors, OpType::ATEN);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);
}

void OpExcuteWatchNormalCheckOpB()
{
    auto &instance = OpExcuteWatch::GetInstance();
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    instance.OpExcuteBegin(stream, "first", OpType::ATB);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    MonitoredTensor tesor1 = {};
    tesor1.data = reinterpret_cast<void *>(DATA_VALUE_1);
    tesor1.dataSize = DATA_SIZE_1;

    MonitoredTensor tesor2 = {};
    tesor2.data = reinterpret_cast<void *>(DATA_VALUE_2);
    tesor2.dataSize = DATA_SIZE_2;

    std::vector<MonitoredTensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.OpExcuteEnd(stream, "first", tensors, OpType::ATB);

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);
}

void OpExcuteWatchNormalCheckOpC()
{
    auto &instance = OpExcuteWatch::GetInstance();
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    EXPECT_EQ(instance.IsInMonitoring(), true);

    instance.OpExcuteBegin(stream, "C", OpType::ATB);

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);

    MonitoredTensor tesor1 = {};
    tesor1.data = reinterpret_cast<void *>(DATA_VALUE_1);
    tesor1.dataSize = DATA_SIZE_1;

    MonitoredTensor tesor2 = {};
    tesor2.data = reinterpret_cast<void *>(DATA_VALUE_2);
    tesor2.dataSize = DATA_SIZE_2;

    std::vector<MonitoredTensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.OpExcuteEnd(stream, "C", tensors, OpType::ATB);

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);
}

void OpExcuteWatchNormalCheckOpD()
{
    auto &instance = OpExcuteWatch::GetInstance();
    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);

    instance.OpExcuteBegin(stream, "last", OpType::ATB);

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);

    MonitoredTensor tesor1 = {};
    tesor1.data = reinterpret_cast<void *>(DATA_VALUE_1);
    tesor1.dataSize = DATA_SIZE_1;

    MonitoredTensor tesor2 = {};
    tesor2.data = reinterpret_cast<void *>(DATA_VALUE_2);
    tesor2.dataSize = DATA_SIZE_2;

    std::vector<MonitoredTensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.OpExcuteEnd(stream, "last", tensors, OpType::ATB);

    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);
}

void OpExcuteWatchNormalCheckOpE()
{
    auto &instance = OpExcuteWatch::GetInstance();
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    instance.OpExcuteBegin(stream, "E", OpType::ATB);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    MonitoredTensor tesor1 = {};
    tesor1.data = reinterpret_cast<void *>(DATA_VALUE_1);
    tesor1.dataSize = DATA_SIZE_1;

    MonitoredTensor tesor2 = {};
    tesor2.data = reinterpret_cast<void *>(DATA_VALUE_2);
    tesor2.dataSize = DATA_SIZE_2;

    std::vector<MonitoredTensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.OpExcuteEnd(stream, "E", tensors, OpType::ATB);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);
}

// 分别执行ABCDE五个OP，其中B和D为start和end算子，校验整个过程中系统的状态是否符合预期
TEST(OpExcuteWatch, OpExcuteWatchNormalCase)
{
    auto &instance = OpExcuteWatch::GetInstance();
    instance.fistWatchOp_ = "first";
    instance.lastWatchOp_ = "last";
    instance.outputId_ = UINT32_MAX;

    OpExcuteWatchNormalCheckOpA();
    OpExcuteWatchNormalCheckOpB();
    OpExcuteWatchNormalCheckOpC();
    OpExcuteWatchNormalCheckOpD();
    OpExcuteWatchNormalCheckOpE();
}

void KernelExcuteWatchNormalCheckOpA()
{
    auto &instance = OpExcuteWatch::GetInstance();
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
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

    instance.KernelExcute(stream, name, tensors, OpType::ATB);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    name = "/A/after";
    instance.KernelExcute(stream, name, tensors, OpType::ATB);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);
}

void KernelExcuteWatchNormalCheckOpB()
{
    auto &instance = OpExcuteWatch::GetInstance();
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
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

    instance.KernelExcute(stream, name, tensors, OpType::ATB);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    name = "/first/after";
    instance.KernelExcute(stream, name, tensors, OpType::ATB);

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);
}

void KernelExcuteWatchNormalCheckOpC()
{
    auto &instance = OpExcuteWatch::GetInstance();
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
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

    instance.KernelExcute(stream, name, tensors, OpType::ATB);

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);

    name = "/C/after";
    instance.KernelExcute(stream, name, tensors, OpType::ATB);

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);
}

void KernelExcuteWatchNormalCheckOpD()
{
    auto &instance = OpExcuteWatch::GetInstance();
    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
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

    instance.KernelExcute(stream, name, tensors, OpType::ATB);

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);

    name = "/last/after";
    instance.KernelExcute(stream, name, tensors, OpType::ATB);

    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);
}

void KernelExcuteWatchNormalCheckOpE()
{
    auto &instance = OpExcuteWatch::GetInstance();
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
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

    instance.KernelExcute(stream, name, tensors, OpType::ATB);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    name = "/E/after";
    instance.KernelExcute(stream, name, tensors, OpType::ATB);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);
}

// 分别执行ABCDE五个kernel，其中B和D为start和end算子，校验整个过程中系统的状态是否符合预期
TEST(OpExcuteWatch, KernelExcuteWatchNormalCase)
{
    auto &instance = OpExcuteWatch::GetInstance();
    instance.fistWatchOp_ = "first";
    instance.lastWatchOp_ = "last";
    instance.outputId_ = UINT32_MAX;

    KernelExcuteWatchNormalCheckOpA();
    KernelExcuteWatchNormalCheckOpB();
    KernelExcuteWatchNormalCheckOpC();
    KernelExcuteWatchNormalCheckOpD();
    KernelExcuteWatchNormalCheckOpE();
}

// 分别执行ABCDE五个算子（OP和kernel混合），其中B和D为start和end算子，且D和E为kernel，其余为OP
// 校验整个过程中系统的状态是否符合预期
TEST(OpExcuteWatch, OpKernelMixExcuteWatchNormalCase)
{
    auto &instance = OpExcuteWatch::GetInstance();
    instance.fistWatchOp_ = "first";
    instance.lastWatchOp_ = "last";
    instance.outputId_ = UINT32_MAX;

    OpExcuteWatchNormalCheckOpA();
    OpExcuteWatchNormalCheckOpB();
    KernelExcuteWatchNormalCheckOpC();
    KernelExcuteWatchNormalCheckOpD();
    OpExcuteWatchNormalCheckOpE();
}

void OpExcuteWatchNormalCheckOpBWithOutputId()
{
    auto &instance = OpExcuteWatch::GetInstance();
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    instance.OpExcuteBegin(stream, "first", OpType::ATB);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);

    MonitoredTensor tesor1 = {};
    tesor1.data = reinterpret_cast<void *>(DATA_VALUE_1);
    tesor1.dataSize = DATA_SIZE_1;

    MonitoredTensor tesor2 = {};
    tesor2.data = reinterpret_cast<void *>(DATA_VALUE_2);
    tesor2.dataSize = DATA_SIZE_2;

    std::vector<MonitoredTensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.OpExcuteEnd(stream, "first", tensors, OpType::ATB);

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);
}

void OpExcuteWatchNormalCheckOpCWithOutputId()
{
    auto &instance = OpExcuteWatch::GetInstance();
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    EXPECT_EQ(instance.IsInMonitoring(), true);

    instance.OpExcuteBegin(stream, "C", OpType::ATB);

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);

    MonitoredTensor tesor1 = {};
    tesor1.data = reinterpret_cast<void *>(DATA_VALUE_3);
    tesor1.dataSize = DATA_SIZE_3;

    MonitoredTensor tesor2 = {};
    tesor2.data = reinterpret_cast<void *>(DATA_VALUE_4);
    tesor2.dataSize = DATA_SIZE_4;

    std::vector<MonitoredTensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.OpExcuteEnd(stream, "C", tensors, OpType::ATB);

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);
}

void OpExcuteWatchNormalCheckOpDWithOutputId()
{
    auto &instance = OpExcuteWatch::GetInstance();
    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);

    instance.OpExcuteBegin(stream, "last", OpType::ATB);

    EXPECT_EQ(instance.GetWatchedOpName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.IsInMonitoring(), true);

    MonitoredTensor tesor1 = {};
    tesor1.data = reinterpret_cast<void *>(DATA_VALUE_5);
    tesor1.dataSize = DATA_SIZE_5;

    MonitoredTensor tesor2 = {};
    tesor2.data = reinterpret_cast<void *>(DATA_VALUE_6);
    tesor2.dataSize = DATA_SIZE_6;

    std::vector<MonitoredTensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.OpExcuteEnd(stream, "last", tensors, OpType::ATB);

    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedOpName(), "");
    EXPECT_EQ(instance.IsInMonitoring(), false);
}

// 分别执行ABCDE五个OP，其中B和D为start和end算子，校验整个过程中系统的状态是否符合预期（设置outputId）
TEST(OpExcuteWatch, OpExcuteWatchSetOutputIdCase)
{
    auto &instance = OpExcuteWatch::GetInstance();
    instance.fistWatchOp_ = "first";
    instance.lastWatchOp_ = "last";
    instance.outputId_ = 1; // 只取下标为1的tensor

    OpExcuteWatchNormalCheckOpA();
    OpExcuteWatchNormalCheckOpBWithOutputId();
    OpExcuteWatchNormalCheckOpCWithOutputId();
    OpExcuteWatchNormalCheckOpDWithOutputId();
    OpExcuteWatchNormalCheckOpE();
}

TEST(OpExcuteWatch, KernelExcuteWatchCheckAtbKernelExcuteFuc)
{
    auto &instance = OpExcuteWatch::GetInstance();
    instance.fistWatchOp_ = "first";
    instance.lastWatchOp_ = "last";
    instance.outputId_ = UINT32_MAX;

    std::string name = "/add/after1";
    Mki::Tensor tensor = {};
    tensor.dataSize = 123;

    Mki::SVector<Mki::Tensor> tensors = {};
    tensors.push_back(tensor);
    instance.KernelExcute(stream, name, tensors, OpType::ATB);

    std::string firstName = "/first/after";
    instance.KernelExcute(stream, firstName, tensors, OpType::ATB);

    tensors.push_back(tensor);
    instance.outputId_ = 0;
    instance.KernelExcute(stream, firstName, tensors, OpType::ATB);
}

TEST(OpExcuteWatch, KernelExcuteWatchCheckAtbOpExcuteBeginFuc)
{
    auto &instance = OpExcuteWatch::GetInstance();
    instance.fistWatchOp_ = "first";
    instance.lastWatchOp_ = "last";
    instance.outputId_ = 0;

    std::string name = "first";

    MonitoredTensor tesor1 = {};
    tesor1.data = reinterpret_cast<void *>(DATA_VALUE_5);
    tesor1.dataSize = DATA_SIZE_5;

    MonitoredTensor tesor2 = {};
    tesor2.data = reinterpret_cast<void *>(DATA_VALUE_6);
    tesor2.dataSize = DATA_SIZE_6;

    std::vector<MonitoredTensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.OpExcuteEnd(stream, name, tensors, OpType::ATB);
}

TEST(OpExcuteWatch, WatchCheckCleanFileNameFunc)
{
    std::string fileName = "/123/123/";
    CleanFileName(fileName);
    EXPECT_EQ(fileName, ".123.123.");
}
