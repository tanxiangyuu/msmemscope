// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <Python.h>
#include <frameobject.h>
#include "utility/call_stack.h"

using namespace Utility;

PyObject *Py_CompileStringExFlags(const char *str, const char *p, int s, PyCompilerFlags *flags, int optimize)
    __attribute__((weak));
void PyErr_Print() __attribute__((weak));
void Py_Finalize() __attribute__((weak));
void Py_Initialize() __attribute__((weak));
PyObject *PyDict_New() __attribute__((weak));
PyObject *PyEval_EvalCode(PyObject *, PyObject *, PyObject *) __attribute__((weak));
int Py_IsInitialized(void) __attribute__((weak));
PyGILState_STATE PyGILState_Ensure(void) __attribute__((weak));
PyFrameObject *PyEval_GetFrame(void) __attribute__((weak));
PyCodeObject* PyFrame_GetCode(PyFrameObject *) __attribute__((weak));
void Py_IncRef(PyObject *) __attribute__((weak));
void Py_DecRef(PyObject *) __attribute__((weak));
PyObject *PyObject_Str(PyObject *) __attribute__((weak));
const char *PyUnicode_AsUTF8(PyObject *) __attribute__((weak));
int PyFrame_GetLineNumber(PyFrameObject*) __attribute__((weak));
PyFrameObject* PyFrame_GetBack(PyFrameObject *) __attribute__((weak));
void PyGILState_Release(PyGILState_STATE) __attribute__((weak));
PyObject *PyImport_ImportModule(const char *name) __attribute__((weak));
void PyErr_Clear(void) __attribute__((weak));
PyObject *PyObject_GetAttrString(PyObject *v, const char *name) __attribute__((weak));
int PyCallable_Check(PyObject *) __attribute__((weak));
PyObject *PyObject_CallObject(PyObject *callable, PyObject *args) __attribute__((weak));

TEST(Call_Stack, get_c_call_stack_test)
{
    std::string stack;
    GetCCallstack(0, stack, 0);
    GetCCallstack(10, stack, 10);
}

TEST(Call_Stack, get_python_call_stack_test)
{
    if (Py_Initialize != nullptr && Py_CompileStringExFlags != nullptr && PyErr_Print != nullptr &&
        Py_Finalize != nullptr && PyDict_New != nullptr && PyEval_EvalCode != nullptr) {
        std::string stack;
        Py_Initialize();
        PyGILState_STATE gstate = PyGILState_Ensure();
        const char* script = "print(123)\n";
        PyObject *code = Py_CompileStringExFlags(script, "<script>", Py_file_input, NULL, -1);
        if (!code) {
            PyErr_Print();
            PyGILState_Release(gstate);
            Py_Finalize();
        }
        PyObject* globals = PyDict_New();
        PyObject* locals = PyDict_New();
        PyEval_EvalCode(code, globals, locals);
        GetPythonCallstack(0, stack);
        GetPythonCallstack(10, stack);
        if (globals != nullptr) {
            Py_DecRef(globals);
        }
        if (locals != nullptr) {
            Py_DecRef(locals);
        }
        if (code != nullptr) {
            Py_DecRef(code);
        }
        PyGILState_Release(gstate);
        Py_Finalize();
    }
}
