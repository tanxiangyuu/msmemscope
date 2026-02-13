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

#include "recordfuncobject.h"
#include "../event_trace/python_trace.h"
#include "../event_trace/trace_manager/event_trace_manager.h"
#include "cpython.h"
#include "utils.h"

namespace MemScope {

const size_t MAX_RECORD_FUNCTION_LENGTH = 128;

// 声明对应的实例化单例模式类（static）
static PyObject* PyMemScopeNewRecordFunction(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    if (type == nullptr || type->tp_alloc == nullptr) {
        return nullptr;
    }

    static PyObject *self = nullptr;
    if (self == nullptr) {
        self = type->tp_alloc(type, 0);
    }

    Py_XINCREF(self);
    return self;
}

// 对应函数接口的具体实现
PyDoc_STRVAR(RecordStartDoc,
"record_start($self, funcname)\n--\n\nEnable debug.");
static PyObject* PyMemScopeRecordStart(PyObject *self,  PyObject *arg)
{
    if(!EventTraceManager::Instance().IsTracingEnabled()) {
        Py_RETURN_NONE;
    }
    if (!PyUnicode_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "[msmemscope] Error: Expected a string argument");
        return nullptr;
    }
    std::string str = Utility::PythonObject(arg).Cast<std::string>();
    if (str.size() > MAX_RECORD_FUNCTION_LENGTH) {
        PyErr_Format(PyExc_ValueError, "[msmemscope] Error: Input funcname exceeds maximum allowed length %zu.", MAX_RECORD_FUNCTION_LENGTH);
        Py_RETURN_NONE;
    }
    PythonTrace::GetInstance().RecordFuncPyCall(str, str, 0);
    Py_RETURN_NONE;
}

PyDoc_STRVAR(RecordEndDoc,
"record_end($self, funcname)\n--\n\nEnable debug.");
static PyObject* PyMemScopeRecordEnd(PyObject *self,  PyObject *arg)
{
    if(!EventTraceManager::Instance().IsTracingEnabled()) {
        Py_RETURN_NONE;
    }
    if (!PyUnicode_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "[msmemscope] Error: Expected a string argument");
        return nullptr;
    }
    std::string str = Utility::PythonObject(arg).Cast<std::string>();
    if (str.size() > MAX_RECORD_FUNCTION_LENGTH) {
        PyErr_Format(PyExc_ValueError, "[msmemscope] Error: Input funcname exceeds maximum allowed length %zu.", MAX_RECORD_FUNCTION_LENGTH);
        Py_RETURN_NONE;
    }
    PythonTrace::GetInstance().RecordFuncReturn(str, str);
    Py_RETURN_NONE;
}

// 对应函数接口的声明
static PyMethodDef PyMemScopeRecordFunctionMethods[] = {
    {"record_start", reinterpret_cast<PyCFunction>(PyMemScopeRecordStart), METH_O, RecordStartDoc},
    {"record_end", reinterpret_cast<PyCFunction>(PyMemScopeRecordEnd), METH_O, RecordEndDoc},
    {nullptr, nullptr, 0, nullptr}
};

// 声明cpython的相关属性
static PyTypeObject PyMemScopeRecordFunctionType = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    "_msmemscope._record_function",                            /* tp_name */
    0,                                                /* tp_basicsize */
    0,                                                /* tp_itemsize */
    /* methods */
    nullptr,                                          /* tp_dealloc */
    0,                                                /* tp_vectorcall_offset */
    nullptr,                                          /* tp_getattr */
    nullptr,                                          /* tp_setattr */
    nullptr,                                          /* tp_as_async */
    nullptr,                                          /* tp_repr */
    nullptr,                                          /* tp_as_number */
    nullptr,                                          /* tp_as_sequence */
    nullptr,                                          /* tp_as_mapping */
    nullptr,                                          /* tp_hash */
    nullptr,                                          /* tp_call */
    nullptr,                                          /* tp_str */
    nullptr,                                          /* tp_getattro */
    nullptr,                                          /* tp_setattro */
    nullptr,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                               /* tp_flags */
    nullptr,                                          /* tp_doc */
    nullptr,                                          /* tp_traverse */
    nullptr,                                          /* tp_clear */
    nullptr,                                          /* tp_richcompare */
    0,                                                /* tp_weaklistoffset */
    nullptr,                                          /* tp_iter */
    nullptr,                                          /* tp_iternext */
    PyMemScopeRecordFunctionMethods,                          /* tp_methods */
    nullptr,                                          /* tp_members */
    nullptr,                                          /* tp_getset */
    &PyBaseObject_Type,                               /* tp_base */
    nullptr,                                          /* tp_dict */
    nullptr,                                          /* tp_descr_get */
    nullptr,                                          /* tp_descr_set */
    0,                                                /* tp_dictoffset */
    nullptr,                                          /* tp_init */
    nullptr,                                          /* tp_alloc */
    PyMemScopeNewRecordFunction,                              /* tp_new */
    PyObject_Del,                                     /* tp_free */
};

// 暴露外部接口
PyObject* PyMemScope_GetRecordFunction()
{
    if (PyType_Ready(&PyMemScopeRecordFunctionType) < 0) {
        return nullptr;
    }

    return PyObject_New(PyObject, &PyMemScopeRecordFunctionType);
}
}