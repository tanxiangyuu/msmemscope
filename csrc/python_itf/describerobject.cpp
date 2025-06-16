// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "describerobject.h"
#include "describe_trace.h"
#include "cpython.h"
#include "utils.h"

namespace Leaks {

/* 单例类，自定义new函数，避免重复构造 */
static PyObject* PyLeaksNewDescriber(PyTypeObject *type, PyObject *args, PyObject *kwds)
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

PyDoc_STRVAR(DescribeDoc,
"describe($self, owner)\n--\n\nEnable debug.");
static PyObject* PyLeaksDescribe(PyObject *self,  PyObject *arg)
{
    if (!PyUnicode_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Expected a string argument");
        return nullptr;
    }
    std::string str = Utility::PythonObject(arg).Cast<std::string>();
    DescribeTrace::GetInstance().AddDescribe(str);
    Py_RETURN_NONE;
}

PyDoc_STRVAR(UnDescribeDoc,
"undescribe($self, owner)\n--\n\nEnable debug.");
static PyObject* PyLeaksUnDescribe(PyObject *self,  PyObject *arg)
{
    if (!PyUnicode_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Expected a string argument");
        return nullptr;
    }
    std::string str = Utility::PythonObject(arg).Cast<std::string>();
    DescribeTrace::GetInstance().EraseDescribe(str);
    Py_RETURN_NONE;
}

PyDoc_STRVAR(DescribeAddrDoc,
"describe_addr($self, addr, owner)\n--\n\nEnable debug.");
static PyObject* PyLeaksDescribeAddr(PyObject *self,  PyObject *args)
{
    unsigned long long addr = 0;
    const char* str = nullptr;

    if (!PyArg_ParseTuple(args, "Ks", &addr, &str)) {
        return NULL;
    }
    DescribeTrace::GetInstance().DescribeAddr(addr, std::string(str));
    Py_RETURN_NONE;
}

static PyMethodDef PyLeaksDescriberMethods[] = {
    {"describe_addr", reinterpret_cast<PyCFunction>(PyLeaksDescribeAddr), METH_VARARGS, DescribeAddrDoc},
    {"describe", reinterpret_cast<PyCFunction>(PyLeaksDescribe), METH_O, DescribeDoc},
    {"undescribe", reinterpret_cast<PyCFunction>(PyLeaksUnDescribe), METH_O, UnDescribeDoc},
    {nullptr, nullptr, 0, nullptr}
};


static PyTypeObject PyLeaksDescriberType = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    "_msleaks._describer",                      /* tp_name */
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
    PyLeaksDescriberMethods,                      /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    &PyBaseObject_Type,                         /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    PyLeaksNewDescriber,                          /* tp_new */
    PyObject_Del,                               /* tp_free */
};

PyObject* PyLeaks_GetDescriber()
{
    if (PyType_Ready(&PyLeaksDescriberType) < 0) {
        return nullptr;
    }

    return PyObject_New(PyObject, &PyLeaksDescriberType);
}
}