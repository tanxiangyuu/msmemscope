// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "process.h"
#include <unistd.h>
#include <sys/wait.h>
#include <iostream>

namespace Leaks {

Process::Process()
{
    analysisFunc_ = nullptr;
    server_ = std::unique_ptr<RemoteProcess>(new RemoteProcess(CommType::SOCKET));
    server_->Start();
}

void Process::Launch(const std::vector<std::string> &execParams)
{
    ::pid_t pid = ::fork();
    if (pid == -1) {
        std::cout << "fork fail" << std::endl;
        return;
    }
    
    if (pid == 0) {
        DoLaunch(execParams); // 执行被检测程序
    } else {
        PostProcess(); // 等待子进程结束，期间不断将client发回的数据转发给分析模块
    }

    return;
}

void Process::RegisterAnalysisFuc(const ANALYSIS_FUNC& analysisFunc)
{
    analysisFunc_ = analysisFunc;
    return;
}

void Process::DoLaunch(const std::vector<std::string> &execParams)
{
    return;
}

void Process::PostProcess()
{
    return;
}

}