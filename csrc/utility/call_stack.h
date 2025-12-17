/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

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