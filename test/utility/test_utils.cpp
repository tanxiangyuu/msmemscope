// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include <gtest/gtest.h>

#include "utility/utils.h"

using namespace Utility;

TEST(Utils, GetAddResult_Overflow_Test)
{
    uint8_t a = 255;
    uint8_t b = 100;
    auto result = GetAddResult(a, b);
    ASSERT_EQ(result, a);
}

TEST(Utils, GetSubResult_Subflow_Test)
{
    uint8_t a = 100;
    uint8_t b = 255;
    auto result = GetSubResult(a, b);
    ASSERT_EQ(result, a);
}