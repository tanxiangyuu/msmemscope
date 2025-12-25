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

