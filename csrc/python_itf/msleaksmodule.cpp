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

PyDoc_STRVAR(StartDoc,
"start()\n--\n\nstart trace data.");
static PyObject* MsleaksStart()
{
    ConfigManager::Instance().InitStartConfig();
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
"config(**kwargs)\n--\n\n"
"Configure msleaks module parameters.\n\n"
"Args:\n"
"    **kwargs: Configuration parameters as keyword arguments\n\n"
"Examples:\n"
"    msleaks.config(call_stack=\"c:10,python:5\", level='0,1')");
static PyObject* MsleaksConfig(PyObject* self, PyObject* args, PyObject* kwargs)
{
    if (PyTuple_Size(args) > 0) {
        PyErr_SetString(PyExc_TypeError, "config() takes no positional arguments");
        return nullptr;
    }
    
    if (!kwargs || PyDict_Size(kwargs) == 0) {
        PyErr_SetString(PyExc_ValueError, "At least one keyword argument is required");
        return nullptr;
    }
    
    std::unordered_map<std::string, std::string> cpp_config;

    PyObject *key;
    PyObject *value;
    Py_ssize_t pos = 0;

    while (PyDict_Next(kwargs, &pos, &key, &value)) {
        if (!PyUnicode_Check(key)) {
            PyErr_SetString(PyExc_TypeError, "Keyword argument names must be strings");
            return nullptr;
        }

        const char* key_str = PyUnicode_AsUTF8(key);
        if (!key_str) {
            return nullptr;
        }

        // 检查值是否为字符串类型（必须加引号）
        if (!PyUnicode_Check(value)) {
            PyErr_Format(PyExc_TypeError, "Value for argument '%s' must be a string (use quotes)", key_str);
            return nullptr;
        }

        const char* value_str = PyUnicode_AsUTF8(value);
        if (!value_str) {
            return nullptr;
        }

        cpp_config.emplace(key_str, value_str);
    }

    bool ret = ConfigManager::Instance().SetConfig(cpp_config);
    if (!ret) {
        PyErr_SetString(PyExc_ValueError, "Set msleaks trace config failed!");
        return nullptr;
    }
    Py_RETURN_NONE;
}

static PyMethodDef g_MsleaksMethods[] = {
    {"start", reinterpret_cast<PyCFunction>(MsleaksStart), METH_NOARGS, StartDoc},
    {"stop", reinterpret_cast<PyCFunction>(MsleaksStop), METH_NOARGS, StopDoc},
    {"config", reinterpret_cast<PyCFunction>(MsleaksConfig), METH_VARARGS | METH_KEYWORDS, ConfigDoc},
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