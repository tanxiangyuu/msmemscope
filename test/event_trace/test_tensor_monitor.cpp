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
#include "event_trace/memory_watch/memory_watch.h"
#include "event_trace/memory_watch/tensor_dumper.h"
#include "event_trace/memory_watch/tensor_monitor.h"

using namespace MemScope;

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
