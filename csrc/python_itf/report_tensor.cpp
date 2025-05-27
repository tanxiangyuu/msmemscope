// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "report_tensor.h"

#include <vector>

#include "event_report.h"
#include "cpython.h"
#include "log.h"
#include "utils.h"
#include "record_info.h"
#include "securec.h"

namespace Leaks {

static PyObject* PyLeaksNewReportTensor(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    if (type == nullptr || type->tp_alloc == nullptr) {
        return nullptr;
    }

    static PyObject *self = nullptr;
    if (self == nullptr) {
        self = type->tp_alloc(type, 0);
    }

    Py_XINCREF(self);
    return self;
}

PyDoc_STRVAR(ReportTensorDoc,
"report_tensor($self, input_list)\n--\n\nEnable debug.");
static PyObject* PyLeaksReportTensor(PyObject *self,  PyObject *arg)
{
    static int tupleSize = 2;
    PyObject* input_list;
    int i;
    int listSize;

    if (!PyList_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a list");
        return nullptr;
    }

    input_list = arg;
    listSize = PyList_Size(input_list);

    for (i = 0; i < listSize; i++) {
        PyObject* item = PyList_GetItem(input_list, i);
        uint64_t addr;
        const char *owner;
        if (!PyTuple_Check(item) || PyTuple_Size(item) != tupleSize) {
            PyErr_SetString(PyExc_TypeError, "Each item must be a tuple of (int, str)");
            return nullptr;
        }
        if (!PyArg_ParseTuple(item, "Ls", &addr, &owner)) {
            PyErr_SetString(PyExc_TypeError, "Tuple elements must be (int, str)");
            return nullptr;
        }
        
        AddrInfo info;
        info.type = AddrInfoType::PTA_OPTIMIZER_STEP;
        info.addr = addr;
        if (strncpy_s(info.owner, sizeof(info.owner), owner, sizeof(info.owner) - 1) != EOK) {
            CLIENT_ERROR_LOG("strncpy_s FAILED");
            info.owner[0] = '\0';
        }
        if (!EventReport::Instance(CommType::SOCKET).ReportAddrInfo(info)) {
            CLIENT_ERROR_LOG("Report optimizer step hook info failed.\n");
        }
    }

    Py_RETURN_NONE;
}

static PyMethodDef PyLeaksReportTensorMethods[] = {
    {"report_tensor", reinterpret_cast<PyCFunction>(PyLeaksReportTensor), METH_O, ReportTensorDoc},
    {nullptr, nullptr, 0, nullptr}
};


static PyTypeObject PyLeaksReportTensorType = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    "_msleaks._report_tensor",                  /* tp_name */
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
    PyLeaksReportTensorMethods,                 /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    &PyBaseObject_Type,                         /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    PyLeaksNewReportTensor,                     /* tp_new */
    PyObject_Del,                               /* tp_free */
};

PyObject* PyLeaks_GetReportTensor()
{
    if (PyType_Ready(&PyLeaksReportTensorType) < 0) {
        return nullptr;
    }

    return PyObject_New(PyObject, &PyLeaksReportTensorType);
}
}