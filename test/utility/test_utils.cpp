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

TEST(Utils, GetSubResult_underflow_Test)
{
    uint8_t a = 100;
    uint8_t b = 255;
    auto result = GetSubResult(a, b);
    ASSERT_EQ(result, a);
}
TEST(Utils, GetAddResult_underflow_Test)
{
    int8_t a = -100;
    int8_t b = -127;
    auto result = GetAddResult(a, b);
    ASSERT_EQ(result, a);
}
TEST(Utils, GetSubResult_Overflow_Test)
{
    int8_t a = 100;
    int8_t b = -127;
    auto result = GetSubResult(a, b);
    ASSERT_EQ(result, a);
}

TEST(Utils, StrToInt64_Failed_Test)
{
    std::string str = "";
    int64_t dest = 0;
    auto ret = StrToInt64(dest, str);
    EXPECT_FALSE(ret);

    str = "6.0";
    ret = StrToInt64(dest, str);
    EXPECT_FALSE(ret);
}

TEST(Utils, StrToUInt64_Failed_Test)
{
    std::string str = "";
    uint64_t dest = 0;
    auto ret = StrToUint64(dest, str);
    EXPECT_FALSE(ret);

    str = "6.0";
    ret = StrToUint64(dest, str);
    EXPECT_FALSE(ret);
}

TEST(Utils, StrToUInt32_Failed_Test)
{
    std::string str = "";
    uint32_t dest = 0;
    auto ret = StrToUint32(dest, str);
    EXPECT_FALSE(ret);

    str = "6.0";
    ret = StrToUint32(dest, str);
    EXPECT_FALSE(ret);
}