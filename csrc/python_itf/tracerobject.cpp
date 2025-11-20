// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "tracerobject.h"
#include "python_trace.h"

namespace MemScope {

/* 单例类，自定义new函数，避免重复构造 */
static PyObject* PyMemScopeNewTracer(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    if (type == nullptr || type->tp_alloc == nullptr) {
        return nullptr;
    }

    /* 单例,减少重复构造 */
    static PyObject *self = nullptr;
    if (self == nullptr) {
        self = type->tp_alloc(type, 0);
    }

    Py_XINCREF(self);
    return self;
}

PyDoc_STRVAR(TraceStartDoc,
"start()\n--\n\nstart trace.");
static PyObject* PyMemScopeTracerStart()
{
    PythonTrace::GetInstance().Start();
    Py_RETURN_NONE;
}

PyDoc_STRVAR(TraceStopDoc,
"stop()\n--\n\nstop trace.");
static PyObject* PyMemScopeTracerStop()
{
    PythonTrace::GetInstance().Stop();
    Py_RETURN_NONE;
}

static PyMethodDef PyMemScopeTracerMethods[] = {
    {"start", reinterpret_cast<PyCFunction>(PyMemScopeTracerStart), METH_NOARGS, TraceStartDoc},
    {"stop", reinterpret_cast<PyCFunction>(PyMemScopeTracerStop), METH_NOARGS, TraceStopDoc},
    {nullptr, nullptr, 0, nullptr}
};


static PyTypeObject PyMemScopeTracerType = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    "_msmemscope._tracer",                               /* tp_name */
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
    PyMemScopeTracerMethods,                             /* tp_methods */
    nullptr,                                          /* tp_members */
    nullptr,                                          /* tp_getset */
    &PyBaseObject_Type,                               /* tp_base */
    nullptr,                                          /* tp_dict */
    nullptr,                                          /* tp_descr_get */
    nullptr,                                          /* tp_descr_set */
    0,                                                /* tp_dictoffset */
    nullptr,                                          /* tp_init */
    nullptr,                                          /* tp_alloc */
    PyMemScopeNewTracer,                                 /* tp_new */
    PyObject_Del,                                     /* tp_free */
};

PyObject* PyMemScope_GetTracer()
{
    if (PyType_Ready(&PyMemScopeTracerType) < 0) {
        return nullptr;
    }

    return PyObject_New(PyObject, &PyMemScopeTracerType);
}
}