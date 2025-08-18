// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include <Python.h>
#include <vector>
#include <string>
#include <unordered_map>
#include "watcherobject.h"
#include "tracerobject.h"
#include "describerobject.h"
#include "report_tensor.h"
#include "trace_manager/event_trace_manager.h"

namespace Leaks {

PyDoc_STRVAR(MsleaksCModuleDoc,
"The part of the module msleaks that is implemented in CXX.\n\
 \n\
...");

PyDoc_STRVAR(StepDoc,
"step($self, /)\n--\n\nUpdata step.");
static PyObject* MsleaksStep(PyObject *self)
{
    Py_RETURN_NONE;
}

PyDoc_STRVAR(StartDoc,
"start()\n--\n\nstart trace data.");
static PyObject* MsleaksStart()
{
    EventTraceManager::Instance().SetTraceStatus(EventTraceStatus::IN_TRACING);
    Py_RETURN_NONE;
}

PyDoc_STRVAR(StopDoc,
"stop()\n--\n\nstop trace data.");
static PyObject* MsleaksStop()
{
    EventTraceManager::Instance().SetTraceStatus(EventTraceStatus::NOT_IN_TRACING);
    Py_RETURN_NONE;
}

PyDoc_STRVAR(ConfigDoc,
"config(...)\n--\n\n"
"Configure msleaks module parameters.\n\n"
"Args:\n"
"    <key>=<value> : Configuration parameters\n\n"
"    msleaks.config('--call-stack=c:10,python:5', '--level=0,1')");
static PyObject* MsleaksConfig(PyObject* self, PyObject* args)
{
    if (PyTuple_Size(args) == 0) {
        PyErr_SetString(PyExc_ValueError, "At least one argument is required");
        return nullptr;
    }
    
    std::unordered_map<std::string, std::string> cpp_config;
    // 处理每个输入参数（如 "--call-stack=c:10,python:5"）
    Py_ssize_t nargs = PyTuple_Size(args);
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        PyObject* arg_obj = PyTuple_GetItem(args, i);
        if (!PyUnicode_Check(arg_obj)) {
            PyErr_SetString(PyExc_TypeError, "Arguments must be strings");
            return nullptr;
        }

        const char* arg = PyUnicode_AsUTF8(arg_obj);
        if (!arg) {
            return nullptr;  // Python异常已设置
        }

        // 验证参数格式 (必须 --key=value)
        if (strncmp(arg, "--", 2) != 0) {
            PyErr_Format(PyExc_ValueError, "Argument must start with '--': %s", arg);
            return nullptr;
        }

        // 直接按等号分割键值
        const char* eq_pos = strchr(arg, '=');
        if (eq_pos) {
            std::string key(arg, eq_pos - arg);
            std::string value(eq_pos + 1);
            cpp_config.emplace(std::move(key), std::move(value));
        } else {
            // 无等号的flag参数
            cpp_config.emplace(arg, "true");
        }
    }

    bool ret = ConfigManager::Instance().SetConfig(cpp_config);
    if (!ret) {
        PyErr_SetString(PyExc_ValueError, "Set msleaks trace config failed!");
        return nullptr;
    }
    Py_RETURN_NONE;
}

static PyMethodDef g_MsleaksMethods[] = {
    {"step", reinterpret_cast<PyCFunction>(MsleaksStep), METH_NOARGS, StepDoc},
    {"start", reinterpret_cast<PyCFunction>(MsleaksStart), METH_NOARGS, StartDoc},
    {"stop", reinterpret_cast<PyCFunction>(MsleaksStop), METH_NOARGS, StopDoc},
    {"config", reinterpret_cast<PyCFunction>(MsleaksConfig), METH_VARARGS, ConfigDoc},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef g_MsleaksCModule = {
    PyModuleDef_HEAD_INIT,
    "_msleaks",                   /* m_name */
    MsleaksCModuleDoc,            /* m_doc */
    -1,                           /* m_size */
    g_MsleaksMethods,             /* m_methods */
};

}

PyMODINIT_FUNC PyInit__msleaks(void)
{
    PyObject* m = PyModule_Create(&Leaks::g_MsleaksCModule);
    if (m == nullptr) {
        return nullptr;
    }

    size_t functionNum = 4;
    std::vector<PyObject*> functions{
        Leaks::PyLeaks_GetWatcher(),
        Leaks::PyLeaks_GetTracer(),
        Leaks::PyLeaks_GetDescriber(),
        Leaks::PyLeaks_GetReportTensor(),
    };
    std::vector<std::string> functionNames{
        "_watcher",
        "_tracer",
        "_describer",
        "_report_tensor",
    };

    for (size_t i = 0; i < functionNum; i++) {
        if (functions[i] == nullptr) {
            Py_DECREF(m);
            return nullptr;
        }
        if (PyModule_AddObject(m, functionNames[i].c_str(), functions[i]) < 0) {
            std::string errorInfo = "Failed to bind " + functionNames[i];
            PyErr_SetString(PyExc_ImportError, errorInfo.c_str());
            Py_DECREF(functions[i]);
            Py_DECREF(m);
            return nullptr;
        }
    }

    return m;
}