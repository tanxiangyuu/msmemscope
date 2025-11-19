// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "call_stack.h"
#include "cpython.h"
#include "utils.h"
#include "event_report.h"
#include "config_info.h"

namespace Utility {
    void GetCallstack(MemScope::CallStackString &stack)
    {
        auto config = MemScope::GetConfig();
        if (config.enableCStack) {
            Utility::GetCCallstack(config.cStackDepth, stack.cStack, MemScope::SKIP_DEPTH);
        }
        if (config.enablePyStack) {
            Utility::GetPythonCallstack(config.pyStackDepth, stack.pyStack);
        }
    }

    void GetPythonCallstack(uint32_t pyDepth, std::string& pyStack)
    {
        PythonCallstack(pyDepth, pyStack);
    }

    void GetCCallstack(uint32_t cDepth, std::string& cStack, uint32_t skip)
    {
        cDepth = GetAddResult(cDepth, skip);
        if (cDepth == skip) {
            return;
        }
        void *buffer[cDepth] = {0};

        int numEntries = backtrace(buffer, cDepth);
        char **symbols = backtrace_symbols(buffer, numEntries);
        if (symbols == nullptr) {
            return;
        }
        cStack += "\"";
        for (int i = 0; i < numEntries; i++) {
            if (skip > 0) {
                skip--;
                continue;
            }
            cStack += std::string(symbols[i]) + '\n';
        }
        cStack += "\"";
        free(symbols);
        return;
    }
}  // namespace Utility