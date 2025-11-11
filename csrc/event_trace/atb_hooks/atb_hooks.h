// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef LEAKS_ATB_HOOK_H
#define LEAKS_ATB_HOOK_H
#include <string>
#include "enum2string.h"
#include "vallina_symbol.h"
#include "kernel_hooks/acl_hooks.h"

namespace atb {
using LeaksOriginalRunnerExecuteFunc = atb::Status (*)(atb::Runner*, atb::RunnerVariantPack&);
using LeaksOriginalGetOperationName = std::string (*)(atb::Runner*);
using LeaksOriginalGetSaveTensorDir = std::string (*)(atb::Runner*);
using LeaksOriginalGetExecuteStream = aclrtStream (*)(atb::Runner*, atb::Context *context);
constexpr uint16_t LEAKS_STRING_MAX_LENGTH = 255;
}

namespace Leaks {
struct ATBLibLoader {
    static void *Load(void)
    {
        std::string libName = "libatb.so";
        const char *pathEnv = std::getenv("ATB_HOME_PATH");
        if (!pathEnv || std::string(pathEnv).empty()) {
            std::cout << "[msleaks] Error: Failed to acquire ATB_HOME_PATH environment variable while loading "
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