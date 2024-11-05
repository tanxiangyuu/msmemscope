// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "process.h"
#include <unistd.h>
#include <sys/wait.h>
#include <iostream>
#include "path.h"
#include "ustring.h"
#include "log.h"

namespace Leaks {

using std::string;
using namespace Utility;

ExecCmd::ExecCmd(std::vector<std::string> const &args) : path_{}, argc_{0}, args_{args}
{
    if (args_.empty()) {
        return;
    }

    /// filename to absolute path
    char *absPath = realpath(args[0].c_str(), nullptr);
    if (absPath) {
        path_ = std::string(absPath);
        free(absPath);
    }

    argc_ = static_cast<int>(args.size());
    for (auto &arg : args_) {
        argv_.push_back(const_cast<char *>(arg.data()));
    }
    argv_.push_back(nullptr);
}

std::string const &ExecCmd::ExecPath(void) const
{
    return path_;
}

char *const *ExecCmd::ExecArgv(void) const
{
    return argv_.data();
}

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
        SetPreloadEnv();
        DoLaunch(execParams);  // 执行被检测程序
    } else {
        PostProcess();  // 等待子进程结束，期间不断将client发回的数据转发给分析模块
    }

    return;
}

void Process::RegisterAnalysisFuc(const ANALYSIS_FUNC &analysisFunc)
{
    analysisFunc_ = analysisFunc;
    return;
}

void Process::SetPreloadEnv()
{
    string hookLibDir = "../lib64/";
    char const *preloadPath = getenv("LD_PRELOAD_PATH");
    if (preloadPath != nullptr && !string(preloadPath).empty()) {
        hookLibDir = preloadPath;
    }

    std::vector<string> hookLibNames{"libascend_hal_hook.so"};

    for (string &hookLib : hookLibNames) {
        Path hookLibPath = (Path(hookLibDir) / Path(hookLib)).Resolved();
        if (hookLibPath.Exists()) {
            hookLib = hookLibPath.ToString();
            Utility::LogInfo("Use preload lib [%s]", hookLib.c_str());
        } else {
            Utility::LogError("No such preload lib [%s]", hookLibPath.ToString().c_str());
        }
    }

    // 按照桩使能模式配置LD_PRELOAD环境变量
    string preloadEnv = Utility::Join(hookLibNames.cbegin(), hookLibNames.cend(), ":");
    const string envName = "LD_PRELOAD";
    auto prevLdPreEnv = getenv(envName.c_str());
    if (prevLdPreEnv && !string(prevLdPreEnv).empty()) {
        preloadEnv += ":" + string(prevLdPreEnv);
    }
    setenv(envName.c_str(), preloadEnv.c_str(), 1);
}

void Process::DoLaunch(const std::vector<std::string> &execParams)
{
    // pass all env-vars from global variable "environ"
    ExecCmd cmd(execParams);
    execve(cmd.ExecPath().c_str(), cmd.ExecArgv(), environ);
    _exit(EXIT_FAILURE);
}

void Process::PostProcess()
{
    auto status = int32_t{};
    wait(&status);
    if (WIFEXITED(status)) {
        if (WEXITSTATUS(status) != EXIT_SUCCESS) {
            std::cout << "[leaks] user program exited abnormally" << std::endl;
        }
    } else if (WIFSIGNALED(status)) {
        int sig = WTERMSIG(status);
        std::cout << "[leaks] user program exited by signal: " << sig << std::endl;
    }
    return;
}

}  // namespace Leaks