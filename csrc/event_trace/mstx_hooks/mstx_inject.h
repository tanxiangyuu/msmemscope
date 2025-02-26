// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#ifndef MSLEAKS_MSTX_INJECT_H
#define MSLEAKS_MSTX_INJECT_H

#include <cstdint>

constexpr int MSTX_SUCCESS = 0;
constexpr int MSTX_FAIL = 1;

enum class mstxFuncModule {
    MSTX_API_MODULE_INVALID                 = 0,
    MSTX_API_MODULE_CORE                    = 1,
    MSTX_API_MODULE_CORE_DOMAIN             = 2,
    MSTX_API_MODULE_CORE_MEM                = 3,
    MSTX_API_MODULE_SIZE,                   // end of the enum, new enum items must be added before this
    MSTX_API_MODULE_FORCE_INT               = 0x7fffffff
};

enum class mstxImplCoreMemFuncId {
    MSTX_API_CORE_MEM_INVALID               = 0,
    MSTX_API_CORE_MEMHEAP_REGISTER          = 1,
    MSTX_API_CORE_MEMHEAP_UNREGISTER        = 2,
    MSTX_API_CORE_MEM_REGIONS_REGISTER      = 3,
    MSTX_API_CORE_MEM_REGIONS_UNREGISTER    = 4,
    MSTX_API_CORE_MEM_SIZE,                   // end of the enum, new enum items must be added before this
    MSTX_API_CORE_MEM_FORCE_INT             = 0x7fffffff
};

enum class mstxImplCoreFuncId {
    MSTX_API_CORE_INVALID                   = 0,
    MSTX_API_CORE_MARK_A                    = 1,
    MSTX_API_CORE_RANGE_START_A             = 2,
    MSTX_API_CORE_RANGE_END                 = 3,
    MSTX_API_CORE_SIZE,                   // end of the enum, new enum items must be added before this
    MSTX_API_CORE_FORCE_INT = 0x7fffffff
};

enum class mstxImplCoreDomainFuncId {
    MSTX_API_CORE2_INVALID                 =  0,
    MSTX_API_CORE2_DOMAIN_CREATE_A         =  1,
    MSTX_API_CORE2_DOMAIN_DESTROY          =  2,
    MSTX_API_CORE2_DOMAIN_MARK_A           =  3,
    MSTX_API_CORE2_DOMAIN_RANGE_START_A    =  4,
    MSTX_API_CORE2_DOMAIN_RANGE_END        =  5,
    MSTX_API_CORE2_SIZE,                   // end of the enum, new enum items must be added before this
    MSTX_API_CORE2_FORCE_INT = 0x7fffffff
};

using aclrtStream = void*;
using MstxFuncPointer = void (*)(void);
using MstxFuncTable = MstxFuncPointer**;
using MstxGetModuleFuncTableFunc = int (*)(mstxFuncModule module, MstxFuncTable *outTable, unsigned int *outSize);

namespace Leaks {
void MstxMarkAFunc(const char* msg, aclrtStream stream);
uint64_t MstxRangeStartAFunc(const char* msg, aclrtStream stream);
void  MstxRangeEndFunc(uint64_t id);
}

#endif