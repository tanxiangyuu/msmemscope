// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#pragma once

#include <string>
#include <Python.h>

namespace Utility {

bool IsPyInterpRepeInited();
void PythonCallstack(uint32_t pyDepth, std::string& pyStack);

class PythonObject {
public:
    PythonObject();
    ~PythonObject();
    PythonObject(const PythonObject &obj);
    PythonObject& operator=(const PythonObject &obj);
    operator PyObject*() const {return ptr;}

    /* 获取模块对象 */
    static PythonObject Import(const std::string& name, bool ignore = true);
    /* 用于获取对象属性，相当于python代码中的obj.xx */
    PythonObject Get(const std::string& name, bool ignore = true) const;
    /* 用于调用可调用对象，相当于python代码中的obj()，先实现无参形式，有需要再扩展 */
    PythonObject Call(bool ignore = true);
    bool IsBad() const {return ptr == nullptr;}

protected:
    explicit PythonObject(PyObject* o);
    void SetPtr(PyObject* o);
    PyObject* ptr{nullptr};

private:
    explicit PythonObject(PythonObject &&obj) = delete;
    PythonObject& operator=(PythonObject &&obj) = delete;
};

class PyInterpGuard {
public:
    PyInterpGuard()
    {
        gstate = PyGILState_Ensure();
    }
    ~PyInterpGuard()
    {
        PyGILState_Release(gstate);
    }

private:
    PyGILState_STATE gstate;
};

}
