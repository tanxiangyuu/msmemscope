// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "atb_hooks.h"

extern "C" atb::Status _ZN3atb6Runner7ExecuteERNS_17RunnerVariantPackE(
    atb::Runner* thisPtr,
    atb::RunnerVariantPack& runnerVariantPack)
{
    return atb::LeaksRunnerExecute(thisPtr, runnerVariantPack);
}

// 不调用原函数，原函数功能不与msleaks兼容
extern "C" void _ZN3atb9StoreUtil15SaveLaunchParamEPvRKN3Mki11LaunchParamERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(
    aclrtStream stream,
    const Mki::LaunchParam &launchParam,
    const std::string &dirPath)
{
    atb::LeaksSaveLaunchParam(launchParam, dirPath);
}

// 劫持判断函数，保证SaveLaunchParam函数可被调用
extern "C" bool _ZN3atb5Probe17IsSaveTensorAfterEv()
{
    return true;
}

extern "C" bool _ZN3atb5Probe18IsSaveTensorBeforeEv()
{
    return true;
}

extern "C" bool _ZN3atb5Probe16IsTensorNeedSaveERKSt6vectorIlSaIlEERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(
    const std::vector<int64_t> &ids,
    const std::string &opType)
{
    return true;
}

extern "C" bool _ZN3atb5Probe21IsExecuteCountInRangeEm(const uint64_t executeCount)
{
    return true;
}

// 劫持判断函数，保证path信息被配置
extern "C" bool _ZN3atb5Probe16IsSaveTensorDescEv()
{
    return true;
}