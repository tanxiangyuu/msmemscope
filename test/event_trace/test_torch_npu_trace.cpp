// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include <gtest/gtest.h>
#include "event_trace/torch_npu_trace/torch_npu_trace.h"
#include "record_info.h"

using namespace Leaks;

TEST(ReportTorchNpuMemData, ReportMemoryUsageTest)
{
    MemoryUsage memoryUsage;
    EXPECT_TRUE(ReportTorchNpuMemData(memoryUsage));
}

