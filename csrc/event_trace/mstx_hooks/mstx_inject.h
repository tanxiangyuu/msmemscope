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
#ifndef MSTX_INJECT_H
#define MSTX_INJECT_H

#include <cstdint>
#include <cstddef>
#include "mstx_info.h"
#include "kernel_hooks/acl_hooks.h"

using MstxFuncPointer = void (*)(void);
using MstxFuncTable = MstxFuncPointer**;
using MstxGetModuleFuncTableFunc = int (*)(mstxFuncModule module, MstxFuncTable *outTable, unsigned int *outSize);

namespace MemScope {

constexpr int MSTX_SUCCESS = 0;
constexpr int MSTX_FAIL = 1;

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