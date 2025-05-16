// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <Python.h>
#include <frameobject.h>
#include "utility/call_stack.h"
#include "cpython.h"

using namespace Utility;

constexpr uint8_t PYTHON_CALL_STACK_DEPTH = 10;
void TestCallStack(std::string, std::string, Leaks::PyTraceType type, uint64_t)
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

TEST(Call_Stack, get_python_call_stack_test)
{
    Py_Initialize();
    PyEval_InitThreads();
    PyGILState_STATE gstate = PyGILState_Ensure();
    const char* script =
        "class Shape:\n"
        "    def area(self):\n"
        "        pass\n"
        "\n"
        "class Circle(Shape):\n"
        "    def __init__(self, radius):\n"
        "        self.radius = radius\n"
        "    def area(self):\n"
        "        return self.radius * self.radius\n"
        "\n"
        "c = Circle(5)\n"
        "c.area()\n";
    RegisterTraceCb(TestCallStack);
    int ret = PyRun_SimpleString(script);
    if (ret != 0) {
        PyErr_Print();
    }
    UnRegisterTraceCb();
    PyGILState_Release(gstate);
    Py_Finalize();
}
