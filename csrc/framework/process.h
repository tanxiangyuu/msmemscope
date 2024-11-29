// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef FRAMEWORK_PROCESS_H
#define FRAMEWORK_PROCESS_H

#include <memory>
#include <string>
#include <thread>
#include "host_injection/core/RemoteProcess.h"
#include "config_info.h"

namespace Leaks {
struct ExecCmd {
    explicit ExecCmd(std::vector<std::string> const &args);
    std::string const &ExecPath(void) const;
    char *const *ExecArgv(void) const;

private:
    std::string path_;
    int argc_;
    std::vector<char*> argv_;
    std::vector<std::string> args_;
};

/*
 * Process类主要功能：
 * 1. Launch用于拉起被检测程序进程，并对client传回的数据进行转发
 * 2. 注册分析回调函数
*/
class Process {
public:
    Process();
    ~Process() = default;
    void Launch(const std::vector<std::string> &execParams);
    void RegisterMsgHandlerHook(ClientMsgHandlerHook msgHandler);
private:
    void SetPreloadEnv();
    void DoLaunch(const ExecCmd &cmd);
    void PostProcess(const ExecCmd &cmd);
private:
    std::unique_ptr<RemoteProcess> server_;
};

}

#endif