// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "tracerobject.h"
#include "python_trace.h"

namespace Leaks {

/* 单例类，自定义new函数，避免重复构造 */
static PyObject* PyLeaksNewTracer(PyTypeObject *type, PyObject *args, PyObject *kwds)
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
static PyObject* PyLeaksTracerStart()
{
    PythonTrace::GetInstance().Start();
    Py_RETURN_NONE;
}

PyDoc_STRVAR(TraceStopDoc,
"stop()\n--\n\nstop trace.");
static PyObject* PyLeaksTracerStop()
{
    PythonTrace::GetInstance().Stop();
    Py_RETURN_NONE;
}

static PyMethodDef PyLeaksTracerMethods[] = {
    {"start", reinterpret_cast<PyCFunction>(PyLeaksTracerStart), METH_NOARGS, TraceStartDoc},
    {"stop", reinterpret_cast<PyCFunction>(PyLeaksTracerStop), METH_NOARGS, TraceStopDoc},
    {nullptr, nullptr, 0, nullptr}
};


static PyTypeObject PyLeaksTracerType = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    "_msleaks._tracer",                         /* tp_name */
    0,                                          /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    0,                                          /* tp_dealloc */
    0,                                          /* tp_vectorcall_offset */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_as_async */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    PyLeaksTracerMethods,                       /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    &PyBaseObject_Type,                         /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    PyLeaksNewTracer,                           /* tp_new */
    PyObject_Del,                               /* tp_free */
};

PyObject* PyLeaks_GetTracer()
{
    if (PyType_Ready(&PyLeaksTracerType) < 0) {
        return nullptr;
    }

    return PyObject_New(PyObject, &PyLeaksTracerType);
}
}