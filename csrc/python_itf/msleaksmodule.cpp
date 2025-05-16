// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include <Python.h>
#include <vector>
#include "watcherobject.h"
#include "tracerobject.h"

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

static PyMethodDef g_MsleaksMethods[] = {
    {"step", reinterpret_cast<PyCFunction>(MsleaksStep), METH_NOARGS, StepDoc},
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

    PyObject* watcher = Leaks::PyLeaks_GetWatcher();
    if (watcher == nullptr) {
        Py_DECREF(m);
        return nullptr;
    }
    if (PyModule_AddObject(m, "_watcher", watcher) < 0) {
        PyErr_SetString(PyExc_ImportError, "Failed to bind watcher.");
        Py_DECREF(watcher);
        Py_DECREF(m);
        return nullptr;
    }

    PyObject* tracer = Leaks::PyLeaks_GetTracer();
    if (tracer == nullptr) {
        Py_DECREF(m);
        return nullptr;
    }
    if (PyModule_AddObject(m, "_tracer", tracer) < 0) {
        PyErr_SetString(PyExc_ImportError, "Failed to bind tracer.");
        Py_DECREF(tracer);
        Py_DECREF(m);
        return nullptr;
    }

    return m;
}