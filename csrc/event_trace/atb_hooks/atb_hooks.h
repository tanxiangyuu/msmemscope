// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef LEAKS_ATB_HOOK_H
#define LEAKS_ATB_HOOK_H

#include "enum2string.h"

namespace atb {
using LeaksOriginalRunnerExecuteFunc = atb::Status (*)(atb::Runner*, atb::RunnerVariantPack&);
std::string LeaksGetTensorInfo(const atb::Tensor& tensor);
std::string LeaksGetTensorInfo(const Mki::Tensor& tensor);
void LeaksReportTensors(atb::RunnerVariantPack& runnerVariantPack);
void LeaksReportTensors(const Mki::LaunchParam &launchParam);
void LeaksReportOp(const std::string& name, const std::string& params, bool isStart);
atb::Status LeaksRunnerExecute(atb::Runner* thisPtr, atb::RunnerVariantPack& runnerVariantPack);
void LeaksSaveLaunchParam(const Mki::LaunchParam &launchParam, const std::string &dirPath);
}

#endif