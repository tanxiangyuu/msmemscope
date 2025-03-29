// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "runtime_hooks.h"

#include <cstdint>
#include <elf.h>
#include <vector>
#include <algorithm>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <unordered_map>
#include <map>

#include "event_report.h"
#include "vallina_symbol.h"
#include "serializer.h"
#include "log.h"
#include "record_info.h"
#include "handle_mapping.h"

using namespace Leaks;

namespace Leaks {

KernelLaunchRecord CreateKernelLaunchRecord(uint32_t blockDim, rtStream_t stm, KernelLaunchType type)
{
    auto record = KernelLaunchRecord {};
    int32_t streamId;
    rtGetStreamId(stm, &streamId);
    record.type = type;
    record.blockDim = blockDim;
    record.streamId = streamId;
    return record;
}
}

RTS_API rtError_t rtKernelLaunch(
    const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize, rtSmDesc_t *smDesc, rtStream_t stm)
{
    using RtKernelLaunch = decltype(&rtKernelLaunch);
    auto vallina = VallinaSymbol<RuntimeLibLoader>::Instance().Get<RtKernelLaunch>(__func__);
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__));
        return RT_ERROR_RESERVED;
    }

    rtError_t ret = vallina(stubFunc, blockDim, args, argsSize, smDesc, stm);
    auto record = KernelLaunchRecord {};
    record = CreateKernelLaunchRecord(blockDim, stm, KernelLaunchType::NORMAL);
    auto hdl = HandleMapping::GetInstance().StubHandleMapFind(stubFunc);
    if (!EventReport::Instance(CommType::SOCKET).ReportKernelLaunch(record, hdl)) {
        CLIENT_ERROR_LOG("rtKernelLaunch report FAILED");
    }
    return ret;
}

RTS_API rtError_t rtKernelLaunchWithHandleV2(void *hdl, const uint64_t tilingKey, uint32_t blockDim,
    rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stm, const rtTaskCfgInfo_t *cfgInfo)
{
    using RtKernelLaunchWithHandleV2 = decltype(&rtKernelLaunchWithHandleV2);
    auto vallina = VallinaSymbol<RuntimeLibLoader>::Instance().Get<RtKernelLaunchWithHandleV2>(__func__);
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__));
        return RT_ERROR_RESERVED;
    }

    rtError_t ret = vallina(hdl, tilingKey, blockDim, argsInfo, smDesc, stm, cfgInfo);
    auto record = KernelLaunchRecord {};
    record = CreateKernelLaunchRecord(blockDim, stm, KernelLaunchType::HANDLEV2);
    if (!EventReport::Instance(CommType::SOCKET).ReportKernelLaunch(record, hdl)) {
        CLIENT_ERROR_LOG("rtKernelLaunchWithHandleV2 report FAILED");
    }
    return ret;
}

RTS_API rtError_t rtKernelLaunchWithFlagV2(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo,
    rtSmDesc_t *smDesc, rtStream_t stm, uint32_t flags, const rtTaskCfgInfo_t *cfgInfo)
{
    using RtKernelLaunchWithFlagV2 = decltype(&rtKernelLaunchWithFlagV2);
    auto vallina = VallinaSymbol<RuntimeLibLoader>::Instance().Get<RtKernelLaunchWithFlagV2>(__func__);
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__));
        return RT_ERROR_RESERVED;
    }

    rtError_t ret = vallina(stubFunc, blockDim, argsInfo, smDesc, stm, flags, cfgInfo);
    auto record = KernelLaunchRecord {};
    record = CreateKernelLaunchRecord(blockDim, stm, KernelLaunchType::FLAGV2);
    auto hdl = HandleMapping::GetInstance().StubHandleMapFind(stubFunc);
    if (!EventReport::Instance(CommType::SOCKET).ReportKernelLaunch(record, hdl)) {
        CLIENT_ERROR_LOG("rtKernelLaunchWithFlagV2 report FAILED");
    }
    return ret;
}

RTS_API rtError_t rtGetStreamId(rtStream_t stm, int32_t *streamId)
{
    using rtGetStreamId = decltype(&rtGetStreamId);
    auto vallina = VallinaSymbol<RuntimeLibLoader>::Instance().Get<rtGetStreamId>(__func__);
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__));
        return RT_ERROR_RESERVED;
    }
    rtError_t ret = vallina(stm, streamId);
    return ret;
}

RTS_API rtError_t rtFunctionRegister(
    void *binHandle, const void *stubFunc, const char *stubName, const void *kernelInfoExt, uint32_t funcMode)
{
    using RtFunctionRegister = decltype(&rtFunctionRegister);
    auto vallina = VallinaSymbol<RuntimeLibLoader>::Instance().Get<RtFunctionRegister>(__func__);
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__));
        return RT_ERROR_RESERVED;
    }
    rtError_t result = vallina(binHandle, stubFunc, stubName, kernelInfoExt, funcMode);
    HandleMapping::GetInstance().StubHandleMapInsert(stubFunc, binHandle);
    return result;
}

RTS_API rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **hdl)
{
    using RtDevBinaryRegister = decltype(&rtDevBinaryRegister);
    auto vallina = VallinaSymbol<RuntimeLibLoader>::Instance().Get<RtDevBinaryRegister>(__func__);
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__));
        return RT_ERROR_RESERVED;
    }
    
    rtError_t result = vallina(bin, hdl);
    if (result == RT_ERROR_NONE && bin != nullptr && bin->data != nullptr && hdl != nullptr) {
        // register handle bin map
        if (bin->length > MAX_BINARY_SIZE) {
            std::string errorInfo = "Illegal binary size: binary size[" + std::to_string(bin->length)
                                    + "] exceeds max binary size[" + std::to_string(MAX_BINARY_SIZE) + "].";
            CLIENT_ERROR_LOG(errorInfo);
            return RT_ERROR_MEMORY_ALLOCATION ;
        }
        auto binData = static_cast<char const *>(bin->data);
        BinKernel binKernel {};
        binKernel.bin = std::vector<char>(binData, binData + bin->length);
        HandleMapping::GetInstance().BinKernelMapInsert(*hdl, binKernel);
    }
    return result;
}

RTS_API rtError_t rtRegisterAllKernel(const rtDevBinary_t *bin, void **hdl)
{
    using RtRegisterAllKernel = decltype(&rtRegisterAllKernel);
    auto vallina = VallinaSymbol<RuntimeLibLoader>::Instance().Get<RtRegisterAllKernel>(__func__);
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__));
        return RT_ERROR_RESERVED;
    }
    rtError_t result = vallina(bin, hdl);
    if (result == RT_ERROR_NONE && bin != nullptr && bin->data != nullptr && hdl != nullptr) {
        // register handle bin map
        if (bin->length > MAX_BINARY_SIZE) {
            std::string errorInfo = "Illegal binary size: binary size[" + std::to_string(bin->length)
                                    + "] exceeds max binary size[" + std::to_string(MAX_BINARY_SIZE) + "].";
            CLIENT_ERROR_LOG(errorInfo);
            return RT_ERROR_MEMORY_ALLOCATION ;
        }
        auto binData = static_cast<char const *>(bin->data);
        BinKernel binKernel {};
        binKernel.bin = std::vector<char>(binData, binData + bin->length);
        HandleMapping::GetInstance().BinKernelMapInsert(*hdl, binKernel);
    }
    return result;
}

RTS_API rtError_t rtDevBinaryUnRegister(void *hdl)
{
    using RtDevBinaryUnRegister = decltype(&rtDevBinaryUnRegister);
    auto vallina = VallinaSymbol<RuntimeLibLoader>::Instance().Get<RtDevBinaryUnRegister>(__func__);
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__));
        return RT_ERROR_RESERVED;
    }

    rtError_t result = vallina(hdl);
    if (result == RT_ERROR_NONE) {
        // unregister handle bin map
        HandleMapping::GetInstance().BinKernelMapErase(hdl);
        // unregister stub handle map
        HandleMapping::GetInstance().StubHandleMapErase(hdl);
    }
    return result;
}