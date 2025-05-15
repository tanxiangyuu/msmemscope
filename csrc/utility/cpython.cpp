// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.Py_IsInitialized

#include <Python.h>
#include <frameobject.h>
#include "utils.h"
#include "cpython.h"

extern "C" {
int Py_IsInitialized(void) __attribute__((weak));
PyGILState_STATE PyGILState_Ensure(void) __attribute__((weak));
PyFrameObject *PyEval_GetFrame(void) __attribute__((weak));
PyCodeObject* PyFrame_GetCode(PyFrameObject *) __attribute__((weak));
PyObject *PyObject_Str(PyObject *) __attribute__((weak));
const char *PyUnicode_AsUTF8(PyObject *) __attribute__((weak));
int PyFrame_GetLineNumber(PyFrameObject*) __attribute__((weak));
PyFrameObject* PyFrame_GetBack(PyFrameObject *) __attribute__((weak));
void PyGILState_Release(PyGILState_STATE) __attribute__((weak));
PyObject *PyImport_ImportModule(const char *name) __attribute__((weak));
PyObject *PyImport_GetModule(PyObject *name) __attribute__((weak));
PyObject *PyObject_GetAttrString(PyObject *v, const char *name) __attribute__((weak));
int PyCallable_Check(PyObject *) __attribute__((weak));
PyObject *PyObject_CallObject(PyObject *callable, PyObject *args) __attribute__((weak));
PyObject *PyObject_Call(PyObject *callable, PyObject *args, PyObject *kwargs) __attribute__((weak));
PyObject *PyUnicode_FromString(const char *u) __attribute__((weak));
PyObject *PyEval_GetGlobals(void) __attribute__((weak));
PyObject *PyLong_FromLong(long ival) __attribute__((weak));
PyObject *PyLong_FromUnsignedLong(unsigned long ival) __attribute__((weak));
PyObject *PyFloat_FromDouble(double fval) __attribute__((weak));
double PyFloat_AsDouble(PyObject *op) __attribute__((weak));
PyObject *PyBool_FromLong(long ok) __attribute__((weak));
int PyObject_IsTrue(PyObject *v) __attribute__((weak));
long PyLong_AsLong(PyObject *obj) __attribute__((weak));
unsigned long PyLong_AsUnsignedLong(PyObject *obj) __attribute__((weak));
PyObject *PyDict_GetItemString(PyObject *v, const char *key) __attribute__((weak));
PyObject *PyList_AsTuple(PyObject *v) __attribute__((weak));
int PyType_IsSubtype(PyTypeObject *a, PyTypeObject *b) __attribute__((weak));
const char *Py_GetVersion() __attribute__((weak));
}

namespace Utility {

const Version VER39("3.9.0");
constexpr uint8_t PRE_ALLOC_SIZE = 2048;
Version GetPyVersion()
{
    const char *ver = Py_GetVersion();
    if (ver == nullptr) {
        return Version("-1");
    }
    std::string version(ver);
    size_t pos = version.find(' ');
    return Version(version.substr(0, pos));
}

bool IsPyInterpRepeInited()
{
    if (Py_IsInitialized != nullptr && Py_IsInitialized()) {
        return true;
    }
    return false;
}

PyInterpGuard::PyInterpGuard()
{
    gstate = PyGILState_Ensure();
}

PyInterpGuard::~PyInterpGuard()
{
    PyGILState_Release(gstate);
}

void PythonCallstack(uint32_t pyDepth, std::string& pyStack)
{
    if (!IsPyInterpRepeInited()) {
        pyStack = "\"NA\"";
        return;
    }
    PyInterpGuard stat{};
    static Version version = GetPyVersion();
    if (version < VER39) {
        pyStack = "\"NA\"";
        return;
    }
    PyFrameObject *frame = PyEval_GetFrame();
    if (frame == nullptr) {
        return;
    }
    Py_IncRef((PyObject *)frame);
    size_t depth = 0;
    pyStack.reserve(PRE_ALLOC_SIZE);
    pyStack += "\"";
    while (frame && depth < pyDepth) {
        PyCodeObject *code = PyFrame_GetCode(frame);
        if (code == nullptr) {
            break;
        }
        PythonObject codeObj(reinterpret_cast<PyObject*>(code));
        auto funcName = codeObj.Get("co_name");
        auto fileName = codeObj.Get("co_filename");
        pyStack += std::string(PyUnicode_AsUTF8(PyObject_Str(fileName))) + "(" +
                   std::to_string(PyFrame_GetLineNumber(frame)) +
                   "): " + std::string(PyUnicode_AsUTF8(PyObject_Str(funcName))) + "\n";

        PyFrameObject *prevFrame = PyFrame_GetBack(frame);
        Py_DecRef((PyObject *)frame);
        frame = prevFrame;
        Py_DecRef((PyObject *)code);
        depth++;
    }
    if (frame != nullptr) {
        Py_DecRef((PyObject *)frame);
    }
    pyStack += "\"";
    return;
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

PythonObject::PythonObject(const PyObject* o)
{
    SetPtr(const_cast<PyObject*>(o));
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

PythonObject::PythonObject(const int32_t& input)
    : PythonObject(static_cast<PyObject*>(PythonNumberObject(input))) {};
PythonObject::PythonObject(const uint32_t& input)
    : PythonObject(static_cast<PyObject*>(PythonNumberObject(input))) {};
PythonObject::PythonObject(const double& input)
    : PythonObject(static_cast<PyObject*>(PythonNumberObject(input))) {};
PythonObject::PythonObject(const std::string& input)
    : PythonObject(static_cast<PyObject*>(PythonStringObject(input))) {};
PythonObject::PythonObject(const char* input)
    : PythonObject(static_cast<PyObject*>(PythonStringObject(input))) {};
PythonObject::PythonObject(const bool& input)
    : PythonObject(static_cast<PyObject*>(PythonBoolObject(input))) {};

PythonObject& PythonObject::NewRef()
{
    if (!IsBad()) {
        Py_IncRef(ptr);
    }
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

template <>
PyObject* PythonObject::Cast<PyObject*>()
{
    return ptr;
}

template <>
PythonObject PythonObject::Cast<PythonObject>()
{
    return *this;
}

template <>
int32_t PythonObject::Cast<int32_t>()
{
    if (IsBad() || !PyLong_Check(ptr)) {
        throw std::runtime_error("Unsupported type conversion");
    }
    return static_cast<int32_t>(PyLong_AsLong(ptr));
}

template <>
uint32_t PythonObject::Cast<uint32_t>()
{
    if (IsBad() || !PyLong_Check(ptr)) {
        throw std::runtime_error("Unsupported type conversion");
    }
    return static_cast<uint32_t>(PyLong_AsUnsignedLong(ptr));
}

template <>
double PythonObject::Cast<double>()
{
    if (IsBad() || !IsInstance("float")) {
        throw std::runtime_error("Unsupported type conversion");
    }
    return static_cast<double>(PyFloat_AsDouble(ptr));
}

template <>
bool PythonObject::Cast<bool>()
{
    if (IsBad() || !PyObject_IsTrue(ptr)) {
        return false;
    }
    return true;
}

template <>
std::string PythonObject::Cast<std::string>()
{
    if (IsBad()) {
        return std::string();
    }
    PyObject* strObj = PyObject_Str(ptr);
    if (strObj == nullptr) {
        return std::string();
    }
    const char* s = PyUnicode_AsUTF8(strObj);
    if (s == nullptr) {
        Py_DecRef(strObj);
        return std::string();
    }

    std::string ret = std::string(s);
    Py_DecRef(strObj);
    return ret;
}

bool PythonObject::IsInstance(const PythonObject& type) const
{
    if (IsBad() || type.IsBad()) {
        return false;
    }

    if (!PyType_Check(static_cast<PyObject*>(type))) {
        return false;
    }

    return PyObject_TypeCheck(ptr, reinterpret_cast<PyTypeObject*>(type.ptr));
}

bool PythonObject::IsInstance(const std::string& type) const
{
    static PythonObject builtin;
    if (builtin.IsBad()) {
        builtin = PythonObject::Import("builtins");
    }
    PythonObject pytype = builtin.Get(type);
    return IsInstance(pytype);
}

bool PythonObject::IsCallable() const
{
    return PyCallable_Check(ptr);
}

std::string PythonObject::Type() const
{
    if (IsBad()) {
        return std::string();
    }

    return std::string(ptr->ob_type->tp_name);
}

PythonObject PythonObject::Import(const std::string& name, bool fromcache, bool ignore)
{
    if (!IsPyInterpRepeInited()) {
        return PythonObject();
    }

    PyObject* m = nullptr;
    if (fromcache) {
        PythonObject pyname(name);
        m = PyImport_GetModule(pyname);
    } else {
        m = PyImport_ImportModule(name.c_str());
    }
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

PythonObject PythonObject::GetGlobal(const std::string& name, bool ignore)
{
    PyObject *globals = PyEval_GetGlobals();
    if (globals == nullptr) {
        if (ignore) {
            PyErr_Clear();
        }
        return PythonObject();
    }

    return PythonObject(PyDict_GetItemString(globals, name.c_str()));
}

PythonObject PythonObject::Get(const std::string& name, bool ignore) const
{
    if (ptr == nullptr) {
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

PythonObject PythonObject::GetItem(const PythonObject& index, bool ignore) const
{
    if (ptr == nullptr || index.ptr == nullptr) {
        return PythonObject();
    }

    do {
        PyObject* getitem = PyObject_GetAttrString(ptr, "__getitem__");
        if (getitem == nullptr) {
            break;
        }

        PyObject* args = PyTuple_New(1);
        if (args == nullptr) {
            Py_DecRef(getitem);
            break;
        }

        Py_IncRef(index);
        PyTuple_SetItem(args, 0, index);

        PyObject* ret = PyObject_CallObject(getitem, args);
        if (ret == nullptr) {
            Py_DecRef(getitem);
            Py_DecRef(args);
            break;
        }

        return PythonObject(ret);
    } while (0);

    /* 走到这个分支说明异常了 */
    if (ignore) {
        PyErr_Clear();
    }
    return PythonObject();
}

PythonObject PythonObject::Call(bool ignore)
{
    if (ptr == nullptr) {
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

PythonObject PythonObject::Call(PythonObject& args, bool ignore)
{
    if (ptr == nullptr) {
        return PythonObject();
    }
    if (!PyCallable_Check(ptr)) {
        return PythonObject();
    }

    if (!PyTuple_Check(args)) {
        return PythonObject();
    }

    PyObject* o = PyObject_CallObject(ptr, args);
    if (o == nullptr && ignore) {
        PyErr_Clear();
    }
    PythonObject ret(o);
    Py_DecRef(o);
    return ret;
}

PythonObject PythonObject::Call(PythonObject& args, PythonObject& kwargs, bool ignore)
{
    if (ptr == nullptr) {
        return PythonObject();
    }
    if (!PyCallable_Check(ptr)) {
        return PythonObject();
    }

    if (!PyTuple_Check(args) || PyDict_Check(kwargs)) {
        return PythonObject();
    }

    PyObject* o = PyObject_Call(ptr, args, kwargs);
    if (o == nullptr && ignore) {
        PyErr_Clear();
    }
    PythonObject ret(o);
    Py_DecRef(o);
    return ret;
}

PythonNumberObject::PythonNumberObject()
{
    PyObject* o = PyLong_FromLong(0);
    SetPtr(o);
    if (o != nullptr) {
        Py_DecRef(o);
    }
}

PythonNumberObject::PythonNumberObject(PyObject* o)
{
    /* PyFloat_Check需要访问PyFloat_Type结构体，此处不要直接使用，需要用isinstance函数 */
    if (o == nullptr || (!PyLong_Check(o) && !PythonObject(o).IsInstance("float"))) {
        return;
    }
    SetPtr(o);
}

PythonNumberObject::PythonNumberObject(const int32_t& input)
{
    PyObject* o = PyLong_FromLong(input);
    SetPtr(o);
    if (o != nullptr) {
        Py_DecRef(o);
    }
}

PythonNumberObject::PythonNumberObject(const uint32_t& input)
{
    PyObject* o = PyLong_FromUnsignedLong(input);
    SetPtr(o);
    if (o != nullptr) {
        Py_DecRef(o);
    }
}

PythonNumberObject::PythonNumberObject(const double& input)
{
    PyObject* o = PyFloat_FromDouble(input);
    SetPtr(o);
    if (o != nullptr) {
        Py_DecRef(o);
    }
}

PythonStringObject::PythonStringObject()
{
    PyObject* o = PyUnicode_FromString("");
    SetPtr(o);
    if (o != nullptr) {
        Py_DecRef(o);
    }
}

PythonStringObject::PythonStringObject(PyObject* o)
{
    if (o == nullptr || !PyUnicode_Check(o)) {
        return;
    }
    SetPtr(o);
}

PythonStringObject::PythonStringObject(const std::string& input)
{
    PyObject* o = PyUnicode_FromString(input.c_str());
    SetPtr(o);
    if (o != nullptr) {
        Py_DecRef(o);
    }
}

PythonStringObject::PythonStringObject(const char* input)
{
    PyObject* o = PyUnicode_FromString(input);
    SetPtr(o);
    if (o != nullptr) {
        Py_DecRef(o);
    }
}

PythonBoolObject::PythonBoolObject()
{
    PyObject* o = PyBool_FromLong(0);
    SetPtr(o);
    if (o != nullptr) {
        Py_DecRef(o);
    }
}

PythonBoolObject::PythonBoolObject(PyObject* o)
{
    PyObject* ret = PyObject_IsTrue(o) ? PyBool_FromLong(1) : PyBool_FromLong(0);
    SetPtr(ret);
    if (ret != nullptr) {
        Py_DecRef(ret);
    }
}

PythonBoolObject::PythonBoolObject(const bool& input)
{
    PyObject* o = PyBool_FromLong(input);
    SetPtr(o);
    if (o != nullptr) {
        Py_DecRef(o);
    }
}

PythonListObject::PythonListObject()
{
    PyObject* o = PyList_New(0);
    SetPtr(o);
    if (o != nullptr) {
        Py_DecRef(o);
    }
}

PythonListObject::PythonListObject(PyObject* o)
{
    if (o == nullptr || !PyList_Check(o)) {
        return;
    }
    SetPtr(o);
}

PythonListObject::PythonListObject(size_t size)
{
    PyObject* o = PyList_New(size);
    SetPtr(o);
    if (o != nullptr) {
        Py_DecRef(o);
    }
}

size_t PythonListObject::Size() const
{
    if (IsBad()) {
        return 0;
    }
    return PyList_Size(ptr);
}

PythonTupleObject PythonListObject::ToTuple(bool ignore)
{
    if (IsBad()) {
        return PythonTupleObject(static_cast<PyObject*>(nullptr));
    }

    PyObject* o = PyList_AsTuple(ptr);
    if (o == nullptr && ignore) {
        PyErr_Clear();
    }

    PythonTupleObject ret(o);
    if (o != nullptr) {
        Py_DecRef(o);
    }
    return ret;
}

PythonTupleObject::PythonTupleObject()
{
    PyObject* o = PyTuple_New(0);
    SetPtr(o);
    if (o != nullptr) {
        Py_DecRef(o);
    }
}

PythonTupleObject::PythonTupleObject(PyObject* o)
{
    if (o == nullptr || !PyTuple_Check(o)) {
        return;
    }
    SetPtr(o);
}

size_t PythonTupleObject::Size() const
{
    if (IsBad()) {
        return 0;
    }
    return PyTuple_Size(ptr);
}

PythonDictObject::PythonDictObject()
{
    PyObject* o = PyDict_New();
    SetPtr(o);
    if (o != nullptr) {
        Py_DecRef(o);
    }
}

PythonDictObject::PythonDictObject(PyObject* o)
{
    if (o == nullptr || !PyDict_Check(o)) {
        return;
    }
    SetPtr(o);
}

}