// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef FRAMEWORK_PROCESS_H
#define FRAMEWORK_PROCESS_H

#include <memory>
#include <string>
#include "host_injection/core/RemoteProcess.h"
#include "config_info.h"

namespace Leaks {
/*
 * Process类主要功能：
 * 1. Launch用于拉起被检测程序进程，并对client传回的数据进行转发
 * 2. 注册分析回调函数
*/
class Process {
public:
    using ANALYSIS_FUNC = std::function<void(std::string)>;
    Process();
    void Launch(const std::vector<std::string> &execParams);
    void RegisterAnalysisFuc(const ANALYSIS_FUNC& analysisFunc);
private:
    void DoLaunch(const std::vector<std::string> &execParams);
    void PostProcess();
private:
    ANALYSIS_FUNC analysisFunc_;
    std::unique_ptr<RemoteProcess> server_;
};

}

#endif