// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "event_trace/hal_hooks/hal_hooks.h"

#include <gtest/gtest.h>

TEST(HalHooksTest, GetMallocMemTypeTest) {
    unsigned long long flag = 2377900603261207558;
    Leaks::MemOpSpace result = GetMemOpSpace(flag);
    EXPECT_EQ(result, Leaks::MemOpSpace::DEVICE);
}


TEST(HalHooksTest, HalMemAllocTest) {
    void* ptr = nullptr;               // 指针初始化为空指针
    unsigned long long size = 1024;    // 分配的大小
    unsigned long long flag = 0x1234;  // 测试 flag

    drvError_t result = halMemAlloc(&ptr, size, flag);
    EXPECT_EQ(result, DRV_ERROR_NONE);

    halMemFree(ptr);
}

// 3. 测试 halMemFree 函数
TEST(HalHooksTest, HalMemFreeTest) {
    void* ptr = malloc(1024); // 手动分配内存，以便测试 halMemFree

    drvError_t result = halMemFree(ptr);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}