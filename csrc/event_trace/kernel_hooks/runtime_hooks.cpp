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

#include "memory_watch/memory_watch.h"
#include "bit_field.h"

using namespace Leaks;

static thread_local bool g_isInAclrtFunc = false;

static void StartKernelEventTrace()
{
    static std::once_flag flag;
    std::call_once(flag, &KernelEventTrace::StartKernelEventTrace, &KernelEventTrace::GetInstance());
}

static void KernelWatchEnd()
{
    bool isKernelLevel = BitPresent(GetConfig().levelType, static_cast<size_t>(LevelType::LEVEL_KERNEL));
    if (!GetConfig().watchConfig.isWatched || !isKernelLevel) {
        return ;
    }
    auto kernelName = RuntimeKernelLinker::GetInstance().GetLastKernelName(Utility::GetTid());
    MemoryWatch::GetInstance().KernelExcuteEnd(nullptr, kernelName);
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

    if (g_isInAclrtFunc) {
        return vallina(stubFunc, blockDim, args, argsSize, smDesc, stm);
    }

    StartKernelEventTrace();
    RuntimeKernelLinker::GetInstance().KernelLaunch();
    rtError_t ret = vallina(stubFunc, blockDim, args, argsSize, smDesc, stm);
    KernelWatchEnd();
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

    if (g_isInAclrtFunc) {
        return vallina(hdl, tilingKey, blockDim, argsInfo, smDesc, stm, cfgInfo);
    }

    StartKernelEventTrace();
    RuntimeKernelLinker::GetInstance().KernelLaunch();
    rtError_t ret = vallina(hdl, tilingKey, blockDim, argsInfo, smDesc, stm, cfgInfo);
    KernelWatchEnd();
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

    if (g_isInAclrtFunc) {
        return vallina(stubFunc, blockDim, argsInfo, smDesc, stm, flags, cfgInfo);
    }

    StartKernelEventTrace();
    RuntimeKernelLinker::GetInstance().KernelLaunch();
    rtError_t ret = vallina(stubFunc, blockDim, argsInfo, smDesc, stm, flags, cfgInfo);
    KernelWatchEnd();
    return ret;
}

RTS_API rtError_t rtAicpuKernelLaunchExWithArgs(const uint32_t kernelType, const char* const opName,
    const uint32_t blockDim, const RtAicpuArgsExT *argsInfo, RtSmDescT * const smDesc, const RtStreamT stm,
    const uint32_t flags)
{
    using RtAicpuKernelLaunchExWithArgs = decltype(&rtAicpuKernelLaunchExWithArgs);
    auto vallina = VallinaSymbol<RuntimeLibLoader>::Instance().Get<RtAicpuKernelLaunchExWithArgs>(__func__);
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__));
        return RT_ERROR_RESERVED;
    }

    if (g_isInAclrtFunc) {
        return vallina(kernelType, opName, blockDim, argsInfo, smDesc, stm, flags);
    }

    StartKernelEventTrace();
    RuntimeKernelLinker::GetInstance().KernelLaunch();
    rtError_t ret = vallina(kernelType, opName, blockDim, argsInfo, smDesc, stm, flags);
    KernelWatchEnd();
    return ret;
}

RTS_API rtError_t rtLaunchKernelByFuncHandle(rtFuncHandle funcHandle, uint32_t blockDim,
    rtLaunchArgsHandle argsHandle, RtStreamT stm)
{
    using RtLaunchKernelByFuncHandle = decltype(&rtLaunchKernelByFuncHandle);
    auto vallina = VallinaSymbol<RuntimeLibLoader>::Instance().Get<RtLaunchKernelByFuncHandle>(__func__);
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__));
        return RT_ERROR_RESERVED;
    }

    if (g_isInAclrtFunc) {
        return vallina(funcHandle, blockDim, argsHandle, stm);
    }

    StartKernelEventTrace();
    RuntimeKernelLinker::GetInstance().KernelLaunch();
    rtError_t ret = vallina(funcHandle, blockDim, argsHandle, stm);
    KernelWatchEnd();
    return ret;
}

RTS_API rtError_t rtLaunchKernelByFuncHandleV2(rtFuncHandle funcHandle, uint32_t blockDim,
    rtLaunchArgsHandle argsHandle, RtStreamT stm, const RtTaskCfgInfoT *cfgInfo)
{
    using RtLaunchKernelByFuncHandleV2 = decltype(&rtLaunchKernelByFuncHandleV2);
    auto vallina = VallinaSymbol<RuntimeLibLoader>::Instance().Get<RtLaunchKernelByFuncHandleV2>(__func__);
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__));
        return RT_ERROR_RESERVED;
    }

    if (g_isInAclrtFunc) {
        return vallina(funcHandle, blockDim, argsHandle, stm, cfgInfo);
    }

    StartKernelEventTrace();
    RuntimeKernelLinker::GetInstance().KernelLaunch();
    rtError_t ret = vallina(funcHandle, blockDim, argsHandle, stm, cfgInfo);
    KernelWatchEnd();
    return ret;
}

aclError aclrtLaunchKernelImpl(aclrtFuncHandle funcHandle, uint32_t blockDim, const void *argsData, size_t argsSize,
                               aclrtStream stream)
{
    StartKernelEventTrace();
    using AclrtLaunchKernel = decltype(&aclrtLaunchKernelImpl);
    static auto vallina = VallinaSymbol<ACLImplLibLoader>::Instance().Get<AclrtLaunchKernel>(__func__);
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__));
        return ACL_ERROR_RT_FAILURE;
    }

    RuntimeKernelLinker::GetInstance().KernelLaunch();
    g_isInAclrtFunc = true;
    aclError ret = vallina(funcHandle, blockDim, argsData, argsSize, stream);
    g_isInAclrtFunc = false;
    KernelWatchEnd();
    return ret;
}

aclError aclrtLaunchKernelWithConfigImpl(aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream,
                                         aclrtLaunchKernelCfg *cfg, aclrtArgsHandle argsHandle, void *reserve)
{
    StartKernelEventTrace();
    using AclrtLaunchKernelWithCfg = decltype(&aclrtLaunchKernelWithConfigImpl);
    static auto vallina = VallinaSymbol<ACLImplLibLoader>::Instance().Get<AclrtLaunchKernelWithCfg>(__func__);
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__));
        return ACL_ERROR_RT_FAILURE;
    }

    RuntimeKernelLinker::GetInstance().KernelLaunch();
    g_isInAclrtFunc = true;
    aclError ret = vallina(funcHandle, blockDim, stream, cfg, argsHandle, reserve);
    g_isInAclrtFunc = false;
    KernelWatchEnd();
    return ret;
}

aclError aclrtLaunchKernelV2Impl(aclrtFuncHandle funcHandle, uint32_t blockDim, const void *argsData, size_t argsSize,
                                 aclrtLaunchKernelCfg *cfg, aclrtStream stream)
{
    StartKernelEventTrace();
    using AclrtLaunchKernelV2 = decltype(&aclrtLaunchKernelV2Impl);
    static auto vallina = VallinaSymbol<ACLImplLibLoader>::Instance().Get<AclrtLaunchKernelV2>(__func__);
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__));
        return ACL_ERROR_RT_FAILURE;
    }

    RuntimeKernelLinker::GetInstance().KernelLaunch();
    g_isInAclrtFunc = true;
    aclError ret = vallina(funcHandle, blockDim, argsData, argsSize, cfg, stream);
    g_isInAclrtFunc = false;
    KernelWatchEnd();
    return ret;
}

