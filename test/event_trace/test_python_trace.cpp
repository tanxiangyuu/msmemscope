// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include <gtest/gtest.h>
#include <unordered_map>
#include "python_trace.h"

using namespace Leaks;

TEST(PythonTrace, RecordPyCallTest)
{
    PythonTrace::GetInstance().RecordPyCall("123", "123", 0);
    PythonTrace::GetInstance().RecordPyCall("123:__torch_dispatch__", "123", 0);
}

TEST(PythonTrace, RecordCCallTest)
{
    PythonTrace::GetInstance().RecordCCall("123", "123");
    PythonTrace::GetInstance().RecordReturn("123:__torch_dispatch__", "123");
}

TEST(PythonTrace, RecordReturnTest)
{
    PythonTrace::GetInstance().Start();
    PythonTrace::GetInstance().RecordReturn("123", "123");
    PythonTrace::GetInstance().RecordPyCall("123", "123", 0);
    PythonTrace::GetInstance().RecordReturn("123", "123");

    std::string hash;
    std::string info;
    uint64_t timestamp = 0;
    PyTraceType what = PyTraceType::PYCALL;
    callback(hash, info, what, timestamp);
    what = PyTraceType::PYRETURN;
    callback(hash, info, what, timestamp);
    what = PyTraceType::CCALL;
    callback(hash, info, what, timestamp);
    what = PyTraceType::CRETURN;
    callback(hash, info, what, timestamp);
    PythonTrace::GetInstance().Stop();
}