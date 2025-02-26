// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "mstx_inject.h"
#include <iostream>
#include "log.h"
#include "mstx_manager.h"
#include "event_report.h"
#include "kernel_hooks/runtime_hooks.h"

namespace Leaks {
void MstxMarkAFunc(const char* msg, aclrtStream stream)
{
    int32_t streamId;
    rtGetStreamId(stream, &streamId);
    MstxManager::GetInstance().ReportMarkA(msg, streamId);
}

uint64_t MstxRangeStartAFunc(const char* msg, aclrtStream stream)
{
    int32_t streamId;
    rtGetStreamId(stream, &streamId);
    return MstxManager::GetInstance().ReportRangeStart(msg, streamId);
}

void  MstxRangeEndFunc(uint64_t id)
{
    MstxManager::GetInstance().ReportRangeEnd(id);
}
}

using namespace Leaks;

extern "C" int __attribute__((visibility("default"))) InitInjectionMstx(MstxGetModuleFuncTableFunc getFuncTable)
{
    unsigned int outSize = 0;
    MstxFuncTable outTable;
    if (getFuncTable == nullptr ||
        getFuncTable(mstxFuncModule::MSTX_API_MODULE_CORE, &outTable, &outSize) != MSTX_SUCCESS ||
        outTable == nullptr) {
        ClientErrorLog("Failed to call getFuncTable");
        return MSTX_FAIL;
    }

    if (outSize >= static_cast<unsigned int>(mstxImplCoreFuncId::MSTX_API_CORE_MARK_A)) {
        *(outTable[static_cast<unsigned int>(mstxImplCoreFuncId::MSTX_API_CORE_MARK_A)]) =
            reinterpret_cast<MstxFuncPointer>(MstxMarkAFunc);
    }

    if (outSize >= static_cast<unsigned int>(mstxImplCoreFuncId::MSTX_API_CORE_RANGE_START_A)) {
        *(outTable[static_cast<unsigned int>(mstxImplCoreFuncId::MSTX_API_CORE_RANGE_START_A)]) =
            reinterpret_cast<MstxFuncPointer>(MstxRangeStartAFunc);
    }

    if (outSize >= static_cast<unsigned int>(mstxImplCoreFuncId::MSTX_API_CORE_RANGE_END)) {
        *(outTable[static_cast<unsigned int>(mstxImplCoreFuncId::MSTX_API_CORE_RANGE_END)]) =
            reinterpret_cast<MstxFuncPointer>(MstxRangeEndFunc);
    }

    return MSTX_SUCCESS;
}