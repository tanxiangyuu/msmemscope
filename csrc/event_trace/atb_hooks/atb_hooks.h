// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef LEAKS_ATB_HOOK_H
#define LEAKS_ATB_HOOK_H

#include "enum2string.h"

namespace atb {
using LeaksOriginalRunnerExecuteFunc = atb::Status (*)(atb::Runner*, atb::RunnerVariantPack&);
std::string LeaksGetOpParams(atb::Runner* thisPtr, atb::RunnerVariantPack& runnerVariantPack);
std::string LeaksGetKernelParams(const std::string &dirPath);
atb::Status LeaksRunnerExecute(atb::Runner* thisPtr, atb::RunnerVariantPack& runnerVariantPack);
void LeaksSaveLaunchParam(const Mki::LaunchParam &launchParam, const std::string &dirPath);
}

#endif