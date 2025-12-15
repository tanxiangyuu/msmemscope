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

#ifndef MEMORY_POOL_TRACE_BASE_H
#define MEMORY_POOL_TRACE_BASE_H

#include <string>
#include "mstx_hooks/mstx_info.h"

namespace MemScope {

// 通过MSTX监控device内存池抽象基类，不同内存池完成各自的泛化处理
class MemoryPoolTraceBase {
public:
    virtual ~MemoryPoolTraceBase() = default;

    // 内存池分配
    virtual mstxMemHeapHandle_t Allocate(mstxDomainHandle_t domain, mstxMemHeapDesc_t const *desc) = 0;

    // 内存池释放
    virtual void Deallocate(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap) = 0;

    // 二次分配
    virtual void Reallocate(mstxDomainHandle_t domain, mstxMemRegionsRegisterBatch_t const *desc) = 0;

    // 二次释放
    virtual void Release(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const *desc) = 0;

    // 维护domain
    virtual mstxDomainHandle_t CreateDomain(const std::string &domainName) = 0;
    
protected:
    MemoryPoolTraceBase() = default;
};
}
#endif