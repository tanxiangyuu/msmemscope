// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include <gtest/gtest.h>
#include "event_trace/op_watch/op_excute_watch.h"
#include "event_trace/op_watch/tensor_dumper.h"
#include "event_trace/op_watch/tensor_monitor.h"

using namespace Leaks;

TEST(TensorMonitorTEST, add_watch_tensor_expect_success)
{
    MonitoredTensor tensor = {};
    uint64_t ptr = 6666;
    tensor.data = reinterpret_cast<void *>(ptr);
    tensor.dataSize = 100;
    TensorMonitor::GetInstance().AddWatchTensor(tensor);
    EXPECT_EQ(TensorMonitor::GetInstance().GetPythonWatchedTensorsMap().size(), 1);
    TensorMonitor::GetInstance().DeleteWatchTensor(tensor);
}

TEST(TensorMonitorTEST, delete_watch_tensor_expect_success)
{
    MonitoredTensor tensor = {};
    uint64_t ptr = 6666;
    tensor.data = reinterpret_cast<void *>(ptr);
    tensor.dataSize = 100;
    TensorMonitor::GetInstance().DeleteWatchTensor(tensor);
    TensorMonitor::GetInstance().AddWatchTensor(tensor);
    EXPECT_EQ(TensorMonitor::GetInstance().GetPythonWatchedTensorsMap().size(), 1);
    TensorMonitor::GetInstance().DeleteWatchTensor(tensor);
    EXPECT_EQ(TensorMonitor::GetInstance().GetPythonWatchedTensorsMap().size(), 0);
}
