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
        
        RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<AddrInfo>(TLVBlockType::ADDR_OWNER, owner);
        AddrInfo* info = buffer.Cast<AddrInfo>();
        info->addrInfoType = AddrInfoType::PTA_OPTIMIZER_STEP;
        info->addr = addr;
        if (!EventReport::Instance(CommType::SOCKET).ReportAddrInfo(buffer)) {
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
    "_msleaks._report_tensor",                        /* tp_name */
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
    PyLeaksReportTensorMethods,                       /* tp_methods */
    nullptr,                                          /* tp_members */
    nullptr,                                          /* tp_getset */
    &PyBaseObject_Type,                               /* tp_base */
    nullptr,                                          /* tp_dict */
    nullptr,                                          /* tp_descr_get */
    nullptr,                                          /* tp_descr_set */
    0,                                                /* tp_dictoffset */
    nullptr,                                          /* tp_init */
    nullptr,                                          /* tp_alloc */
    PyLeaksNewReportTensor,                           /* tp_new */
    PyObject_Del,                                     /* tp_free */
};

PyObject* PyLeaks_GetReportTensor()
{
    if (PyType_Ready(&PyLeaksReportTensorType) < 0) {
        return nullptr;
    }

    return PyObject_New(PyObject, &PyLeaksReportTensorType);
}
}