// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "runtime_hooks.h"

#include <cstdint>
#include <elf.h>
#include <vector>
#include <algorithm>
#include <sstream>
#include <cstdio>
#include <mutex>
#include <cstdlib>
#include <iterator>
#include <unordered_map>
#include <map>

#include "event_report.h"
#include "vallina_symbol.h"
#include "serializer.h"
#include "log.h"
#include "record_info.h"
#include "kernel_event_trace.h"

using namespace Leaks;

void static StartKernelEventTrace()
{
    static std::once_flag flag;
    std::call_once(flag, &KernelEventTrace::StartKernelEventTrace, &KernelEventTrace::GetInstance());
}

RTS_API rtError_t rtKernelLaunch(
    const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize, rtSmDesc_t *smDesc, rtStream_t stm)
{
    StartKernelEventTrace();
    using RtKernelLaunch = decltype(&rtKernelLaunch);
    auto vallina = VallinaSymbol<RuntimeLibLoader>::Instance().Get<RtKernelLaunch>(__func__);
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__));
        return RT_ERROR_RESERVED;
    }

    RuntimeKernelLinker::GetInstance().KernelLaunch();
    rtError_t ret = vallina(stubFunc, blockDim, args, argsSize, smDesc, stm);
    return ret;
}

RTS_API rtError_t rtKernelLaunchWithHandleV2(void *hdl, const uint64_t tilingKey, uint32_t blockDim,
    rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stm, const rtTaskCfgInfo_t *cfgInfo)
{
    StartKernelEventTrace();
    using RtKernelLaunchWithHandleV2 = decltype(&rtKernelLaunchWithHandleV2);
    auto vallina = VallinaSymbol<RuntimeLibLoader>::Instance().Get<RtKernelLaunchWithHandleV2>(__func__);
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__));
        return RT_ERROR_RESERVED;
    }

    RuntimeKernelLinker::GetInstance().KernelLaunch();
    rtError_t ret = vallina(hdl, tilingKey, blockDim, argsInfo, smDesc, stm, cfgInfo);
    return ret;
}

RTS_API rtError_t rtKernelLaunchWithFlagV2(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo,
    rtSmDesc_t *smDesc, rtStream_t stm, uint32_t flags, const rtTaskCfgInfo_t *cfgInfo)
{
    StartKernelEventTrace();
    using RtKernelLaunchWithFlagV2 = decltype(&rtKernelLaunchWithFlagV2);
    auto vallina = VallinaSymbol<RuntimeLibLoader>::Instance().Get<RtKernelLaunchWithFlagV2>(__func__);
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__));
        return RT_ERROR_RESERVED;
    }

    RuntimeKernelLinker::GetInstance().KernelLaunch();
    rtError_t ret = vallina(stubFunc, blockDim, argsInfo, smDesc, stm, flags, cfgInfo);
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