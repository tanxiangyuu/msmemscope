// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef LEAKS_ATB_HOOK_H
#define LEAKS_ATB_HOOK_H
#include <string>
#include "enum2string.h"
#include "vallina_symbol.h"

namespace atb {
using LeaksOriginalRunnerExecuteFunc = atb::Status (*)(atb::Runner*, atb::RunnerVariantPack&);
using LeaksOriginalGetOperationName = std::string (*)(atb::Runner*);
using LeaksOriginalGetSaveTensorDir = std::string (*)(atb::Runner*);
std::string LeaksGetTensorInfo(const atb::Tensor& tensor);
std::string LeaksGetTensorInfo(const Mki::Tensor& tensor);
void LeaksReportTensors(atb::RunnerVariantPack& runnerVariantPack);
void LeaksReportTensors(Mki::LeaksOriginalGetInTensors &getInTensors, Mki::LeaksOriginalGetInTensors &getOutTensors,
    const Mki::LaunchParam &launchParam);
void LeaksReportOp(const std::string& name, const std::string& params, bool isStart);
atb::Status LeaksRunnerExecute(atb::Runner* thisPtr, atb::RunnerVariantPack& runnerVariantPack);
void LeaksSaveLaunchParam(const Mki::LaunchParam &launchParam, const std::string &dirPath);
}

namespace Leaks {
struct ATBLibLoader {
    static void *Load(void)
    {
        return dlopen("libatb.so", RTLD_NOW | RTLD_GLOBAL);
    }
};
}

#endif