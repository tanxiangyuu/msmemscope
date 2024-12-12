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

#include "event_report.h"
#include "vallina_symbol.h"
#include "serializer.h"
#include "log.h"
#include "record_info.h"

using namespace Leaks;

RTS_API rtError_t rtKernelLaunch(
    const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize, rtSmDesc_t *smDesc, rtStream_t stm)
{
    using RtKernelLaunch = decltype(&rtKernelLaunch);
    auto vallina = VallinaSymbol<RuntimeLibLoader>::Instance().Get<RtKernelLaunch>(__func__);
    if (vallina == nullptr) {
        Utility::LogError("vallina func get FAILED");
        return RT_ERROR_RESERVED;
    }

    rtError_t ret = vallina(stubFunc, blockDim, args, argsSize, smDesc, stm);
    if (!EventReport::Instance(CommType::SOCKET).ReportKernelLaunch(KernelLaunchType::NORMAL)) {
        Utility::LogError("%s report FAILED", __func__);
    }

    return ret;
}

RTS_API rtError_t rtKernelLaunchWithHandleV2(void *hdl, const uint64_t tilingKey, uint32_t blockDim,
    rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stm, const rtTaskCfgInfo_t *cfgInfo)
{
    using RtKernelLaunchWithHandleV2 = decltype(&rtKernelLaunchWithHandleV2);
    auto vallina = VallinaSymbol<RuntimeLibLoader>::Instance().Get<RtKernelLaunchWithHandleV2>(__func__);
    if (vallina == nullptr) {
        Utility::LogError("vallina func get FAILED");
        return RT_ERROR_RESERVED;
    }

    rtError_t ret = vallina(hdl, tilingKey, blockDim, argsInfo, smDesc, stm, cfgInfo);
    if (!EventReport::Instance(CommType::SOCKET).ReportKernelLaunch(KernelLaunchType::HANDLEV2)) {
        Utility::LogError("%s report FAILED", __func__);
    }

    return ret;
}

RTS_API rtError_t rtKernelLaunchWithFlagV2(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo,
    rtSmDesc_t *smDesc, rtStream_t stm, uint32_t flags, const rtTaskCfgInfo_t *cfgInfo)
{
    using RtKernelLaunchWithFlagV2 = decltype(&rtKernelLaunchWithFlagV2);
    auto vallina = VallinaSymbol<RuntimeLibLoader>::Instance().Get<RtKernelLaunchWithFlagV2>(__func__);
    if (vallina == nullptr) {
        Utility::LogError("vallina func get FAILED");
        return RT_ERROR_RESERVED;
    }

    rtError_t ret = vallina(stubFunc, blockDim, argsInfo, smDesc, stm, flags, cfgInfo);
    if (!EventReport::Instance(CommType::SOCKET).ReportKernelLaunch(KernelLaunchType::FLAGV2)) {
        Utility::LogError("%s report FAILED", __func__);
    }

    return ret;
}