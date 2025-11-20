// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include <gtest/gtest.h>
#include <unordered_map>
#include "describe_trace.h"

using namespace MemScope;

TEST(DescribeTrace, AddDescribeTest)
{
    DescribeTrace::GetInstance().AddDescribe("123");
    DescribeTrace::GetInstance().AddDescribe("123");
    DescribeTrace::GetInstance().AddDescribe("456");
    DescribeTrace::GetInstance().AddDescribe("789");
    DescribeTrace::GetInstance().AddDescribe("012");
}

TEST(DescribeTrace, EraseDescribeTest)
{
    DescribeTrace::GetInstance().EraseDescribe("123");
    DescribeTrace::GetInstance().AddDescribe("123");
    DescribeTrace::GetInstance().EraseDescribe("456");
}

TEST(PythonTrace, AddAddrDescribeTest)
{
    DescribeTrace::GetInstance().DescribeAddr(123, "123");
}