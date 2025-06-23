// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "describerobject.h"
#include "describe_trace.h"
#include "cpython.h"
#include "utils.h"

namespace Leaks {

const size_t MAX_DESCRIBE_OWNER_LENGTH = 64;

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
    if (str.size() > MAX_DESCRIBE_OWNER_LENGTH) {
        PyErr_Format(PyExc_ValueError, "Input owner exceeds maximum allowed length %zu.", MAX_DESCRIBE_OWNER_LENGTH);
        Py_RETURN_NONE;
    }
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
    if (str.size() > MAX_DESCRIBE_OWNER_LENGTH) {
        PyErr_Format(PyExc_ValueError, "Input owner exceeds maximum allowed length %zu.", MAX_DESCRIBE_OWNER_LENGTH);
        Py_RETURN_NONE;
    }
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
        return nullptr;
    }
    if (std::strlen(str) > MAX_DESCRIBE_OWNER_LENGTH) {
        PyErr_Format(PyExc_ValueError, "Input owner exceeds maximum allowed length %zu.", MAX_DESCRIBE_OWNER_LENGTH);
        Py_RETURN_NONE;
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
    "_msleaks._describer",                            /* tp_name */
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
    PyLeaksDescriberMethods,                          /* tp_methods */
    nullptr,                                          /* tp_members */
    nullptr,                                          /* tp_getset */
    &PyBaseObject_Type,                               /* tp_base */
    nullptr,                                          /* tp_dict */
    nullptr,                                          /* tp_descr_get */
    nullptr,                                          /* tp_descr_set */
    0,                                                /* tp_dictoffset */
    0,                                                /* tp_init */
    0,                                                /* tp_alloc */
    PyLeaksNewDescriber,                              /* tp_new */
    PyObject_Del,                                     /* tp_free */
};

PyObject* PyLeaks_GetDescriber()
{
    if (PyType_Ready(&PyLeaksDescriberType) < 0) {
        return nullptr;
    }

    return PyObject_New(PyObject, &PyLeaksDescriberType);
}
}