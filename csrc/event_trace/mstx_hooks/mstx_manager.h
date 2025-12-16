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
#ifndef MSLEAKS_MSTX_MANAGER_H
#define MSLEAKS_MSTX_MANAGER_H

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <iostream>
#include "record_info.h"
#include "mstx_info.h"
#include "comm_def.h"

namespace MemScope {
/*
* mstx_inject仅仅做接口的转发调用，实际的上报信息组装由Manager来完成，避免接口变动时改动过大
*/
class MstxManager {
public:
    static MstxManager& GetInstance()
    {
        static MstxManager instance;
        return instance;
    }
    MstxManager(const MstxManager&) = delete;
    MstxManager& operator=(const MstxManager&) = delete;

    void ReportMarkA(const char* msg, int32_t streamId, MemScopeCommType type = MemScopeCommType::SHARED_MEMORY);
    uint64_t ReportRangeStart(const char* msg, int32_t streamId);
    void ReportRangeEnd(uint64_t id);
    mstxDomainHandle_t ReportDomainCreateA(char const *domainName);
    mstxMemHeapHandle_t ReportHeapRegister(mstxDomainHandle_t domain, mstxMemHeapDesc_t const *desc);
    void ReportHeapUnregister(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap);
    void ReportRegionsRegister(mstxDomainHandle_t domain, mstxMemRegionsRegisterBatch_t const *desc);
    void ReportRegionsUnregister(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const *desc);

private:
    MstxManager()
    {
        rangeId_ = 1;
    }
    uint64_t GetRangeId();

private:
    std::atomic<uint64_t> rangeId_;
    const int onlyMarkId_ = 0;
};

}

#endif