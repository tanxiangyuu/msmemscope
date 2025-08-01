// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include <Python.h>
#include <vector>
#include <string>
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

static PyMethodDef g_MsleaksMethods[] = {
    {"step", reinterpret_cast<PyCFunction>(MsleaksStep), METH_NOARGS, StepDoc},
    {"start", reinterpret_cast<PyCFunction>(MsleaksStart), METH_NOARGS, StartDoc},
    {"stop", reinterpret_cast<PyCFunction>(MsleaksStop), METH_NOARGS, StopDoc},
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