// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "mstx_inject.h"
#include <iostream>
#include "log.h"
#include "mstx_manager.h"
#include "event_report.h"

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

bool MstxTableCoreInject(MstxGetModuleFuncTableFunc getFuncTable)
{
    unsigned int outSize = 0;
    MstxFuncTable outTable;
    if (getFuncTable == nullptr ||
        getFuncTable(mstxFuncModule::MSTX_API_MODULE_CORE, &outTable, &outSize) != MSTX_SUCCESS ||
        outTable == nullptr) {
        CLIENT_ERROR_LOG("Failed to call getFuncTablecore");
        return false;
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
    return true;
}

bool MstxTableDomainCoreInject(MstxGetModuleFuncTableFunc getFuncTable)
{
    unsigned int outSize = 0;
    MstxFuncTable outTable;
    if (getFuncTable == nullptr ||
        getFuncTable(mstxFuncModule::MSTX_API_MODULE_CORE_DOMAIN, &outTable, &outSize) != MSTX_SUCCESS ||
        outTable == nullptr) {
        CLIENT_ERROR_LOG("Failed to call getFuncTableDomaincore");
        return false;
    }

    if (outSize >= static_cast<unsigned int>(mstxImplCoreDomainFuncId::MSTX_API_CORE2_DOMAIN_CREATE_A)) {
        *(outTable[static_cast<unsigned int>(mstxImplCoreDomainFuncId::MSTX_API_CORE2_DOMAIN_CREATE_A)]) =
        reinterpret_cast<MstxFuncPointer>(MstxDomainCreateAFunc);
    }
    return true;
}

bool MstxTableMemCoreInject(MstxGetModuleFuncTableFunc getFuncTable)
{
    unsigned int outSize = 0;
    MstxFuncTable outTable;
    if (getFuncTable == nullptr ||
        getFuncTable(mstxFuncModule::MSTX_API_MODULE_CORE_MEM, &outTable, &outSize) != MSTX_SUCCESS ||
        outTable == nullptr) {
        CLIENT_ERROR_LOG("Failed to call getFuncTableMemcore");
        return false;
    }

    if (outSize >= static_cast<unsigned int>(mstxImplCoreMemFuncId::MSTX_API_CORE_MEMHEAP_REGISTER)) {
        *(outTable[static_cast<unsigned int>(mstxImplCoreMemFuncId::MSTX_API_CORE_MEMHEAP_REGISTER)]) =
        reinterpret_cast<MstxFuncPointer>(MstxMemHeapRegisterFunc);
    }
    if (outSize >= static_cast<unsigned int>(mstxImplCoreMemFuncId::MSTX_API_CORE_MEMHEAP_UNREGISTER)) {
        *(outTable[static_cast<unsigned int>(mstxImplCoreMemFuncId::MSTX_API_CORE_MEMHEAP_UNREGISTER)]) =
        reinterpret_cast<MstxFuncPointer>(MstxMemHeapUnregisterFunc);
    }
    if (outSize >= static_cast<unsigned int>(mstxImplCoreMemFuncId::MSTX_API_CORE_MEM_REGIONS_REGISTER)) {
        *(outTable[static_cast<unsigned int>(mstxImplCoreMemFuncId::MSTX_API_CORE_MEM_REGIONS_REGISTER)]) =
        reinterpret_cast<MstxFuncPointer>(MstxMemRegionsRegisterFunc);
    }
    if (outSize >= static_cast<unsigned int>(mstxImplCoreMemFuncId::MSTX_API_CORE_MEM_REGIONS_UNREGISTER)) {
        *(outTable[static_cast<unsigned int>(mstxImplCoreMemFuncId::MSTX_API_CORE_MEM_REGIONS_UNREGISTER)]) =
        reinterpret_cast<MstxFuncPointer>(MstxMemRegionsUnregisterFunc);
    }
    
    return true;
}

mstxDomainHandle_t MstxDomainCreateAFunc(char const *domainName)
{
    return MstxManager::GetInstance().ReportDomainCreateA(domainName);
}

mstxMemHeapHandle_t MstxMemHeapRegisterFunc(mstxDomainHandle_t domain, mstxMemHeapDesc_t const *desc)
{
    return MstxManager::GetInstance().ReportHeapRegister(domain, desc);
}

void MstxMemHeapUnregisterFunc(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap)
{
    MstxManager::GetInstance().ReportHeapUnregister(domain, heap);
}

void MstxMemRegionsRegisterFunc(mstxDomainHandle_t domain, mstxMemRegionsRegisterBatch_t const *desc)
{
    MstxManager::GetInstance().ReportRegionsRegister(domain, desc);
}

void MstxMemRegionsUnregisterFunc(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const *desc)
{
    MstxManager::GetInstance().ReportRegionsUnregister(domain, desc);
}

}

using namespace Leaks;

extern "C" int __attribute__((visibility("default"))) InitInjectionMstx(MstxGetModuleFuncTableFunc getFuncTable)
{
    bool isCoreInjection = MstxTableCoreInject(getFuncTable);
    bool isDomainCoreInjection = MstxTableDomainCoreInject(getFuncTable);
    bool isMemCoreInjection = MstxTableMemCoreInject(getFuncTable);
    if (isCoreInjection && isDomainCoreInjection && isMemCoreInjection) {
        return MSTX_SUCCESS;
    }
    return MSTX_FAIL;
}