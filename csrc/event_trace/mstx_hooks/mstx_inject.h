// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#ifndef MSLEAKS_MSTX_INJECT_H
#define MSLEAKS_MSTX_INJECT_H

#include <cstdint>

constexpr int MSTX_SUCCESS = 0;
constexpr int MSTX_FAIL = 1;

enum class MstxFuncSeq {
    MSTX_FUNC_START = 0,
    MSTX_FUNC_MARKA = 1,
    MSTX_FUNC_RANGE_STARTA = 2,
    MSTX_FUNC_RANGE_END = 3,
    MSTX_FUNC_END = 4
};

enum class MstxFuncModule {
    MSTX_API_MODULE_INVALID                 = 0,
    MSTX_API_MODULE_CORE                    = 1,
    MSTX_API_MODULE_SIZE                    = 2,
    MSTX_API_MODULE_FORCE_INT               = 0x7fffffff
};

using aclrtStream = void*;
using MstxFuncPointer = void (*)(void);
using MstxFuncTable = MstxFuncPointer**;
using MstxGetModuleFuncTableFunc = int (*)(MstxFuncModule module, MstxFuncTable *outTable, unsigned int *outSize);

namespace Leaks {
void MstxMarkAFunc(const char* msg, aclrtStream stream);
uint64_t MstxRangeStartAFunc(const char* msg, aclrtStream stream);
void  MstxRangeEndFunc(uint64_t id);
}

#endif