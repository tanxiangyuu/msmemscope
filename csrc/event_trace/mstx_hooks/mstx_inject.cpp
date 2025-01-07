// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "mstx_inject.h"
#include <iostream>
#include "log.h"
#include "mstx_manager.h"
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
        getFuncTable(MstxFuncModule::MSTX_API_MODULE_CORE, &outTable, &outSize) != MSTX_SUCCESS ||
        outTable == nullptr) {
        std::cout << "Failed to call getFuncTable" << std::endl;
        return MSTX_FAIL;
    }

    if (outSize != static_cast<unsigned int>(MstxFuncSeq::MSTX_FUNC_END)) {
        std::cout << "OutSize is not equal to MSTX_FUNC_END, Failed to init mstx funcs." << std::endl;
        return MSTX_FAIL; // 1 : init failed
    }
    *(outTable[static_cast<unsigned int>(MstxFuncSeq::MSTX_FUNC_MARKA)]) =
        reinterpret_cast<MstxFuncPointer>(MstxMarkAFunc);
    *(outTable[static_cast<unsigned int>(MstxFuncSeq::MSTX_FUNC_RANGE_STARTA)]) =
        reinterpret_cast<MstxFuncPointer>(MstxRangeStartAFunc);
    *(outTable[static_cast<unsigned int>(MstxFuncSeq::MSTX_FUNC_RANGE_END)]) =
        reinterpret_cast<MstxFuncPointer>(MstxRangeEndFunc);
    return MSTX_SUCCESS;
}