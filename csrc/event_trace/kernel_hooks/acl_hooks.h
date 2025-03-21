// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef LEAKS_HOOKS_ACL_HOOKS_H
#define LEAKS_HOOKS_ACL_HOOKS_H

#include <cstdint>
#include <cstddef>
#include <dlfcn.h>
#include "vallina_symbol.h"

namespace Leaks {
constexpr int ACL_SUCCESS = 0;
constexpr int ACL_ERROR_INTERNAL_ERROR = 500000;
 
struct AclLibLoader {
    static void *Load(void)
    {
        return LibLoad("libascendcl.so");
    }
};
}

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define ACL_FUNC_VISIBILITY _declspec(dllexport)
#else
#define ACL_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define ACL_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define ACL_FUNC_VISIBILITY
#endif
#endif

typedef int aclError;

ACL_FUNC_VISIBILITY aclError aclInit(const char *configPath);
ACL_FUNC_VISIBILITY aclError aclFinalize();

#ifdef __cplusplus
}
#endif

#endif