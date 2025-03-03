// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#ifndef MSLEAKS_MSTX_INJECT_H
#define MSLEAKS_MSTX_INJECT_H

#include <cstdint>
#include <cstddef>
#include "mstx_info.h"
constexpr int MSTX_SUCCESS = 0;
constexpr int MSTX_FAIL = 1;

using aclrtStream = void*;
using MstxFuncPointer = void (*)(void);
using MstxFuncTable = MstxFuncPointer**;
using MstxGetModuleFuncTableFunc = int (*)(mstxFuncModule module, MstxFuncTable *outTable, unsigned int *outSize);

namespace Leaks {
void MstxMarkAFunc(const char* msg, aclrtStream stream);
uint64_t MstxRangeStartAFunc(const char *msg, aclrtStream stream);
void MstxRangeEndFunc(uint64_t id);

bool MstxTableDomainCoreInject(MstxGetModuleFuncTableFunc getFuncTable);
bool MstxTableCoreInject(MstxGetModuleFuncTableFunc getFuncTable);
bool MstxTableMemCoreInject(MstxGetModuleFuncTableFunc getFuncTable);

mstxDomainHandle_t MstxDomainCreateAFunc(char const *domainName);
mstxMemHeapHandle_t MstxMemHeapRegisterFunc(mstxDomainHandle_t domain, mstxMemHeapDesc_t const *desc);
void MstxMemHeapUnregisterFunc(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap);
void MstxMemRegionsRegisterFunc(mstxDomainHandle_t domain, mstxMemRegionsRegisterBatch_t const *desc);
void MstxMemRegionsUnregisterFunc(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const *desc);
}

#endif