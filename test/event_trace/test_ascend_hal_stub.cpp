// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "event_trace/ascend_hal.h"

#include <gtest/gtest.h>

TEST(AscendHalStubTest, HalMemAllocInnerReturnsSuccess) {
    void* ptr = nullptr;
    unsigned long long size = 1024;
    unsigned long long flag = 0;

    drvError_t result = halMemAllocInner(&ptr, size, flag);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST(AscendHalStubTest, HalMemFreeInnerReturnsSuccess) {
    void* ptr = nullptr;

    drvError_t result = halMemFreeInner(ptr);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}