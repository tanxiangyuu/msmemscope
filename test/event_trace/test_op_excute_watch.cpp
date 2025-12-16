/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

#include <gtest/gtest.h>
#define private public
#include "event_trace/memory_watch/memory_watch.h"
#undef private
#include "event_trace/memory_watch/tensor_dumper.h"
#include "event_trace/memory_watch/tensor_monitor.h"
#include "securec.h"

using namespace MemScope;

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

class MemoryWatchTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        Utility::FileCreateManager::GetInstance("./testmsmemscope").SetProjectDir("./testmsmemscope");
    }

    void TearDown() override
    {
        Utility::FileCreateManager::GetInstance("./testmsmemscope").SetProjectDir("");
        rmdir("./testmsmemscope");
    }
};

void OpExcuteWatchNormalCheckOpA()
{
    auto &instance = MemoryWatch::GetInstance();
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedTargetName(), "");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), false);

    instance.OpExcuteBegin(stream, "A");

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedTargetName(), "");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), false);

    instance.OpExcuteBegin(stream, "A");

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedTargetName(), "");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), false);

    MonitoredTensor tesor1 = {};
    tesor1.data = reinterpret_cast<void *>(DATA_VALUE_1);
    tesor1.dataSize = DATA_SIZE_1;

    MonitoredTensor tesor2 = {};
    tesor2.data = reinterpret_cast<void *>(DATA_VALUE_2);
    tesor2.dataSize = DATA_SIZE_2;

    std::vector<MonitoredTensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.OpExcuteEnd(stream, "A", tensors);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedTargetName(), "");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), false);

    instance.OpExcuteEnd(stream, "A", tensors);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedTargetName(), "");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), false);
}

void OpExcuteWatchNormalCheckOpB()
{
    auto &instance = MemoryWatch::GetInstance();
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedTargetName(), "");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), false);

    instance.OpExcuteBegin(stream, "first");

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedTargetName(), "");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), false);

    MonitoredTensor tesor1 = {};
    tesor1.data = reinterpret_cast<void *>(DATA_VALUE_1);
    tesor1.dataSize = DATA_SIZE_1;

    MonitoredTensor tesor2 = {};
    tesor2.data = reinterpret_cast<void *>(DATA_VALUE_2);
    tesor2.dataSize = DATA_SIZE_2;

    std::vector<MonitoredTensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.OpExcuteEnd(stream, "first", tensors);

    EXPECT_EQ(instance.GetWatchedTargetName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), true);
}

void OpExcuteWatchNormalCheckOpC()
{
    auto &instance = MemoryWatch::GetInstance();
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.GetWatchedTargetName(), "first");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), true);

    instance.OpExcuteBegin(stream, "C");

    EXPECT_EQ(instance.GetWatchedTargetName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), true);

    MonitoredTensor tesor1 = {};
    tesor1.data = reinterpret_cast<void *>(DATA_VALUE_1);
    tesor1.dataSize = DATA_SIZE_1;

    MonitoredTensor tesor2 = {};
    tesor2.data = reinterpret_cast<void *>(DATA_VALUE_2);
    tesor2.dataSize = DATA_SIZE_2;

    std::vector<MonitoredTensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.OpExcuteEnd(stream, "C", tensors);

    EXPECT_EQ(instance.GetWatchedTargetName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), true);
}

void OpExcuteWatchNormalCheckOpD()
{
    auto &instance = MemoryWatch::GetInstance();
    EXPECT_EQ(instance.GetWatchedTargetName(), "first");
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), true);

    instance.OpExcuteBegin(stream, "last");

    EXPECT_EQ(instance.GetWatchedTargetName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), true);

    MonitoredTensor tesor1 = {};
    tesor1.data = reinterpret_cast<void *>(DATA_VALUE_1);
    tesor1.dataSize = DATA_SIZE_1;

    MonitoredTensor tesor2 = {};
    tesor2.data = reinterpret_cast<void *>(DATA_VALUE_2);
    tesor2.dataSize = DATA_SIZE_2;

    std::vector<MonitoredTensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.OpExcuteEnd(stream, "last", tensors);

    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedTargetName(), "");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), false);
}

void OpExcuteWatchNormalCheckOpE()
{
    auto &instance = MemoryWatch::GetInstance();
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedTargetName(), "");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), false);

    instance.OpExcuteBegin(stream, "E");

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedTargetName(), "");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), false);

    MonitoredTensor tesor1 = {};
    tesor1.data = reinterpret_cast<void *>(DATA_VALUE_1);
    tesor1.dataSize = DATA_SIZE_1;

    MonitoredTensor tesor2 = {};
    tesor2.data = reinterpret_cast<void *>(DATA_VALUE_2);
    tesor2.dataSize = DATA_SIZE_2;

    std::vector<MonitoredTensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.OpExcuteEnd(stream, "E", tensors);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedTargetName(), "");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), false);
}

// 分别执行ABCDE五个OP，其中B和D为start和end算子，校验整个过程中系统的状态是否符合预期
TEST_F(MemoryWatchTest, OpExcuteWatchNormalCase)
{
    auto &instance = MemoryWatch::GetInstance();
    instance.firstWatchTarget_ = "first";
    instance.lastWatchTarget_ = "last";
    instance.outputId_ = UINT32_MAX;

    OpExcuteWatchNormalCheckOpA();
    OpExcuteWatchNormalCheckOpB();
    OpExcuteWatchNormalCheckOpC();
    OpExcuteWatchNormalCheckOpD();
    OpExcuteWatchNormalCheckOpE();
}

void KernelExcuteWatchNormalCheckOpA()
{
    auto &instance = MemoryWatch::GetInstance();
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedTargetName(), "");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), false);

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

    instance.ATBKernelExcute(stream, (char *)name.c_str(), tensors);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedTargetName(), "");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), false);

    name = "/A/after";
    instance.ATBKernelExcute(stream, (char *)name.c_str(), tensors);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedTargetName(), "");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), false);
}

void KernelExcuteWatchNormalCheckOpB()
{
    auto &instance = MemoryWatch::GetInstance();
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedTargetName(), "");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), false);

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

    instance.ATBKernelExcute(stream, (char *)name.c_str(), tensors);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedTargetName(), "");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), false);

    name = "/first/after";
    instance.ATBKernelExcute(stream, (char *)name.c_str(), tensors);

    EXPECT_EQ(instance.GetWatchedTargetName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), true);
}

void KernelExcuteWatchNormalCheckOpC()
{
    auto &instance = MemoryWatch::GetInstance();
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.GetWatchedTargetName(), "first");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), true);

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

    instance.ATBKernelExcute(stream, (char *)name.c_str(), tensors);

    EXPECT_EQ(instance.GetWatchedTargetName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), true);

    name = "/C/after";
    instance.ATBKernelExcute(stream, (char *)name.c_str(), tensors);

    EXPECT_EQ(instance.GetWatchedTargetName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), true);
}

void KernelExcuteWatchNormalCheckOpD()
{
    auto &instance = MemoryWatch::GetInstance();
    EXPECT_EQ(instance.GetWatchedTargetName(), "first");
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), true);

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

    instance.ATBKernelExcute(stream, (char *)name.c_str(), tensors);

    EXPECT_EQ(instance.GetWatchedTargetName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].dataSize, DATA_SIZE_1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_1].data, reinterpret_cast<void *>(DATA_VALUE_1));
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), true);

    name = "/last/after";
    instance.ATBKernelExcute(stream, (char *)name.c_str(), tensors);

    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedTargetName(), "");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), false);
}

void KernelExcuteWatchNormalCheckOpE()
{
    auto &instance = MemoryWatch::GetInstance();
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedTargetName(), "");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), false);

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

    instance.ATBKernelExcute(stream, (char *)name.c_str(), tensors);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedTargetName(), "");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), false);

    name = "/E/after";
    instance.ATBKernelExcute(stream, (char *)name.c_str(), tensors);

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedTargetName(), "");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), false);
}

// 分别执行ABCDE五个kernel，其中B和D为start和end算子，校验整个过程中系统的状态是否符合预期
TEST_F(MemoryWatchTest, KernelExcuteWatchNormalCase)
{
    auto &instance = MemoryWatch::GetInstance();
    instance.firstWatchTarget_ = "first";
    instance.lastWatchTarget_ = "last";
    instance.outputId_ = UINT32_MAX;

    KernelExcuteWatchNormalCheckOpA();
    KernelExcuteWatchNormalCheckOpB();
    KernelExcuteWatchNormalCheckOpC();
    KernelExcuteWatchNormalCheckOpD();
    KernelExcuteWatchNormalCheckOpE();
}

// 分别执行ABCDE五个算子（OP和kernel混合），其中B和D为start和end算子，且D和E为kernel，其余为OP
// 校验整个过程中系统的状态是否符合预期
TEST_F(MemoryWatchTest, OpKernelMixExcuteWatchNormalCase)
{
    auto &instance = MemoryWatch::GetInstance();
    instance.firstWatchTarget_ = "first";
    instance.lastWatchTarget_ = "last";
    instance.outputId_ = UINT32_MAX;

    OpExcuteWatchNormalCheckOpA();
    OpExcuteWatchNormalCheckOpB();
    KernelExcuteWatchNormalCheckOpC();
    KernelExcuteWatchNormalCheckOpD();
    OpExcuteWatchNormalCheckOpE();
}

void OpExcuteWatchNormalCheckOpBWithOutputId()
{
    auto &instance = MemoryWatch::GetInstance();
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedTargetName(), "");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), false);

    instance.OpExcuteBegin(stream, "first");

    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedTargetName(), "");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), false);

    MonitoredTensor tesor1 = {};
    tesor1.data = reinterpret_cast<void *>(DATA_VALUE_1);
    tesor1.dataSize = DATA_SIZE_1;

    MonitoredTensor tesor2 = {};
    tesor2.data = reinterpret_cast<void *>(DATA_VALUE_2);
    tesor2.dataSize = DATA_SIZE_2;

    std::vector<MonitoredTensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.OpExcuteEnd(stream, "first", tensors);

    EXPECT_EQ(instance.GetWatchedTargetName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), true);
}

void OpExcuteWatchNormalCheckOpCWithOutputId()
{
    auto &instance = MemoryWatch::GetInstance();
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(instance.GetWatchedTargetName(), "first");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), true);

    instance.OpExcuteBegin(stream, "C");

    EXPECT_EQ(instance.GetWatchedTargetName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), true);

    MonitoredTensor tesor1 = {};
    tesor1.data = reinterpret_cast<void *>(DATA_VALUE_3);
    tesor1.dataSize = DATA_SIZE_3;

    MonitoredTensor tesor2 = {};
    tesor2.data = reinterpret_cast<void *>(DATA_VALUE_4);
    tesor2.dataSize = DATA_SIZE_4;

    std::vector<MonitoredTensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.OpExcuteEnd(stream, "C", tensors);

    EXPECT_EQ(instance.GetWatchedTargetName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), true);
}

void OpExcuteWatchNormalCheckOpDWithOutputId()
{
    auto &instance = MemoryWatch::GetInstance();
    EXPECT_EQ(instance.GetWatchedTargetName(), "first");
    auto watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), true);

    instance.OpExcuteBegin(stream, "last");

    EXPECT_EQ(instance.GetWatchedTargetName(), "first");
    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 1);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].dataSize, DATA_SIZE_2);
    EXPECT_EQ(watchedTensors[DATA_VALUE_2].data, reinterpret_cast<void *>(DATA_VALUE_2));
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), true);

    MonitoredTensor tesor1 = {};
    tesor1.data = reinterpret_cast<void *>(DATA_VALUE_5);
    tesor1.dataSize = DATA_SIZE_5;

    MonitoredTensor tesor2 = {};
    tesor2.data = reinterpret_cast<void *>(DATA_VALUE_6);
    tesor2.dataSize = DATA_SIZE_6;

    std::vector<MonitoredTensor> tensors = {};
    tensors.push_back(tesor1);
    tensors.push_back(tesor2);

    instance.OpExcuteEnd(stream, "last", tensors);

    watchedTensors = TensorMonitor::GetInstance().GetCmdWatchedTensorsMap();
    EXPECT_EQ(watchedTensors.size(), 0);
    EXPECT_EQ(instance.GetWatchedTargetName(), "");
    EXPECT_EQ(TensorMonitor::GetInstance().IsInMonitoring(), false);
}

// 分别执行ABCDE五个OP，其中B和D为start和end算子，校验整个过程中系统的状态是否符合预期（设置outputId）
TEST_F(MemoryWatchTest, OpExcuteWatchSetOutputIdCase)
{
    auto &instance = MemoryWatch::GetInstance();
    instance.firstWatchTarget_ = "first";
    instance.lastWatchTarget_ = "last";
    instance.outputId_ = 1; // 只取下标为1的tensor

    OpExcuteWatchNormalCheckOpA();
    OpExcuteWatchNormalCheckOpBWithOutputId();
    OpExcuteWatchNormalCheckOpCWithOutputId();
    OpExcuteWatchNormalCheckOpDWithOutputId();
    OpExcuteWatchNormalCheckOpE();
}

TEST_F(MemoryWatchTest, KernelExcuteWatchCheckAtbKernelExcuteFuc)
{
    auto &instance = MemoryWatch::GetInstance();
    instance.firstWatchTarget_ = "first";
    instance.lastWatchTarget_ = "last";
    instance.outputId_ = UINT32_MAX;

    std::string name = "/add/after1";
    Mki::Tensor tensor = {};
    tensor.dataSize = 123;

    Mki::SVector<Mki::Tensor> tensors = {};
    tensors.push_back(tensor);
    instance.ATBKernelExcute(stream, (char *)name.c_str(), tensors);

    std::string firstName = "/first/after";
    instance.ATBKernelExcute(stream, (char *)firstName.c_str(), tensors);

    tensors.push_back(tensor);
    instance.outputId_ = 0;
    instance.ATBKernelExcute(stream, (char *)firstName.c_str(), tensors);
}

TEST_F(MemoryWatchTest, KernelExcuteWatchCheckAtbOpExcuteBeginFuc)
{
    auto &instance = MemoryWatch::GetInstance();
    instance.firstWatchTarget_ = "first";
    instance.lastWatchTarget_ = "last";
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

    instance.OpExcuteEnd(stream, name, tensors);
}

TEST_F(MemoryWatchTest, WatchCheckCleanFileNameFunc)
{
    std::string fileName = "/123/123/";
    CleanFileName(fileName);
    EXPECT_EQ(fileName, ".123.123.");
}
