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

#ifndef LEAKS_ATB_HOOK_H
#define LEAKS_ATB_HOOK_H
#include <string>
#include "enum2string.h"
#include "vallina_symbol.h"
#include "kernel_hooks/acl_hooks.h"

namespace atb {
using MemScopeOriginalRunnerExecuteFunc = atb::Status (*)(atb::Runner*, atb::RunnerVariantPack&);
using MemScopeOriginalGetOperationName = std::string (*)(atb::Runner*);
using MemScopeOriginalGetSaveTensorDir = std::string (*)(atb::Runner*);
using MemScopeOriginalGetExecuteStream = aclrtStream (*)(atb::Runner*, atb::Context *context);
constexpr uint16_t LEAKS_STRING_MAX_LENGTH = 255;
}

namespace MemScope {
struct ATBLibLoader {
    static void *Load(void)
    {
        std::string libName = "libatb.so";
        const char *pathEnv = std::getenv("ATB_HOME_PATH");
        if (!pathEnv || std::string(pathEnv).empty()) {
            std::cout << "[msmemscope] Error: Failed to acquire ATB_HOME_PATH environment variable while loading "
                << libName << "." << std::endl;
            return nullptr;
        }
        std::string libPath = pathEnv;
        libPath += "/lib/" + libName;
        return dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    }
};
}

#endif