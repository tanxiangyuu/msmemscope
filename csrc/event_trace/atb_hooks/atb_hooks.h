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
}

namespace Leaks {
struct ATBLibLoader {
    static void *Load(void)
    {
        return dlopen("libatb.so", RTLD_NOW | RTLD_GLOBAL);
    }
};
}

#endif