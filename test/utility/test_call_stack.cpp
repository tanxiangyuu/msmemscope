// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <Python.h>
#include <frameobject.h>
#include "utility/call_stack.h"
#include "cpython.h"

using namespace Utility;

constexpr uint8_t PYTHON_CALL_STACK_DEPTH = 10;
void TestCallStack(const std::string&, const std::string&, MemScope::PyTraceType type, uint64_t)
{
    std::string stack;
    GetPythonCallstack(0, stack);
    GetPythonCallstack(PYTHON_CALL_STACK_DEPTH, stack);
}

TEST(Call_Stack, get_c_call_stack_test)
{
    std::string stack;
    GetCCallstack(0, stack, 0);
    GetCCallstack(10, stack, 10);
}

