// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef CALL_STACK_H
#define CALL_STACK_H

#include <iostream>
#include <string>
#include <Python.h>
#include <frameobject.h>
#include <execinfo.h>
#include "record_info.h"

namespace Utility {
    // 先跳过skip层，再采pyDepth层调用栈,结果保存在入参pyStack里
    void GetPythonCallstack(uint32_t pyDepth, std::string& pyStack);
    void GetCCallstack(uint32_t cDepth, std::string& cStack, uint32_t skip = 0);

    void GetCallstack(MemScope::CallStackString &stack);
}
#endif