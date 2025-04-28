// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.Py_IsInitialized

#include <Python.h>
#include <frameobject.h>
#include "cpython.h"

extern "C" {
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
}

namespace Leaks {

bool IsPyInterpRepeInited()
{
    if (Py_IsInitialized != nullptr && Py_IsInitialized()) {
        return true;
    }
    return false;
}

PythonObject::PythonObject() {}
PythonObject::~PythonObject()
{
    if (ptr != nullptr && Py_DecRef != nullptr) {
        Py_DecRef(ptr);
    }
}

PythonObject::PythonObject(PyObject* o)
{
    SetPtr(o);
}

PythonObject::PythonObject(const PythonObject &obj)
{
    SetPtr(obj.ptr);
}

PythonObject& PythonObject::operator=(const PythonObject &obj)
{
    SetPtr(obj.ptr);
    return *this;
}

void PythonObject::SetPtr(PyObject* o)
{
    if (ptr != nullptr && Py_DecRef != nullptr) {
        Py_DecRef(ptr);
    }
    if (o != nullptr && Py_IncRef != nullptr) {
        Py_IncRef(o);
    }
    ptr = o;
}

PythonObject PythonObject::Import(const std::string& name, bool ignore)
{
    if (!IsPyInterpRepeInited()) {
        return PythonObject();
    }
    PyObject* m = PyImport_ImportModule(name.c_str());
    if (m == nullptr) {
        if (ignore) {
            PyErr_Clear();
        }
        return PythonObject();
    }
    PythonObject ret(m);
    Py_DecRef(m);
    return ret;
}

PythonObject PythonObject::Get(const std::string& name, bool ignore) const
{
    if (!IsPyInterpRepeInited() || ptr == nullptr) {
        return PythonObject();
    }
    PyObject* o = PyObject_GetAttrString(ptr, name.c_str());
    if (o == nullptr) {
        if (ignore) {
            PyErr_Clear();
        }
        return PythonObject();
    }
    PythonObject ret(o);
    Py_DecRef(o);
    return ret;
}

PythonObject PythonObject::Call(bool ignore)
{
    if (!IsPyInterpRepeInited() || ptr == nullptr) {
        return PythonObject();
    }
    if (!PyCallable_Check(ptr)) {
        return PythonObject();
    }

    PyObject* o = PyObject_CallObject(ptr, nullptr);
    if (o == nullptr) {
        if (ignore) {
            PyErr_Clear();
        }
        return PythonObject();
    }
    PythonObject ret(o);
    Py_DecRef(o);
    return ret;
}

}