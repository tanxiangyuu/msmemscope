// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "process.h"
#include <unistd.h>
#include <sys/wait.h>
#include <iostream>
#include "path.h"
#include "ustring.h"
#include "log.h"
#include "serializer.h"
#include "analysis/dump_record.h"
#include "analysis/trace_record.h"
#include "analysis/mstx_analyzer.h"
#include "analysis/hal_analyzer.h"
#include "analysis/stepinner_analyzer.h"

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
    } else {
        path_ = args[0];
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

Process::Process(const AnalysisConfig &config)
{
    config_ = config;
    server_ = std::unique_ptr<ServerProcess>(new ServerProcess(CommType::SOCKET));

    server_->SetClientConnectHook([this, config](ClientId clientId) {
            this->server_->Notify(clientId, Serialize<AnalysisConfig>(config));
        });

    auto func = std::bind(&Process::MsgHandle, this, std::placeholders::_1, std::placeholders::_2);
    server_->SetMsgHandlerHook(func);

    server_->Start();
}

void Process::RecordHandler(const ClientId &clientId, const EventRecord &record)
{
    DumpRecord::GetInstance().DumpData(clientId, record);
    TraceRecord::GetInstance().TraceHandler(record);

    switch (record.type) {
        case RecordType::MSTX_MARK_RECORD:
            MstxAnalyzer::Instance().RecordMstx(clientId, record.record.mstxRecord);
            break;
        case RecordType::MEMORY_RECORD:
            HalAnalyzer::GetInstance().Record(clientId, record);
            break;
        case RecordType::TORCH_NPU_RECORD:
            StepInnerAnalyzer::GetInstance(config_).Record(clientId, record);
            break;
        default:
            break;
    }

    return;
}

void Process::MsgHandle(size_t &clientId, std::string &msg)
{
    if (protocolList_.find(clientId) == protocolList_.end()) {
        auto result = protocolList_.insert({clientId, Protocol{}});
        if (!result.second) {
            LOG_ERROR("Add elements to protocolList failed, clientId = %u", clientId);
        }
    }

    Protocol protocol = protocolList_[clientId];
    protocol.Feed(msg);

    while (true) {
        auto packet = protocol.GetPacket();
        switch (packet.GetPacketHead().type) {
            case PacketType::RECORD:
                RecordHandler(clientId, packet.GetPacketBody().eventRecord);
                break;
            case PacketType::LOG: {
                auto log = packet.GetPacketBody().log;
                LOG_RECV("%s", std::string(log.buf, log.buf + log.len).c_str());
                break;
            }
            case PacketType::INVALID:
            default:
                return;
        }
    }

    return;
}

void Process::Launch(const std::vector<std::string> &execParams)
{
    ExecCmd cmd(execParams);
    ::pid_t pid = ::fork();
    if (pid == -1) {
        std::cout << "fork fail" << std::endl;
        return;
    }

    if (pid == 0) {
        SetPreloadEnv();
        DoLaunch(cmd);  // 执行被检测程序
    } else {
        PostProcess(cmd);  // 等待子进程结束，期间不断将client发回的数据转发给分析模块
    }

    return;
}

void Process::SetPreloadEnv()
{
    string hookLibDir = "../lib64/";
    char const *preloadPath = getenv("LD_PRELOAD_PATH");
    if (preloadPath != nullptr && !string(preloadPath).empty()) {
        hookLibDir = preloadPath;
    }

    std::vector<string> hookLibNames{
        "libascend_hal_hook.so",
        "libhost_memory_hook.so",
        "libascend_mstx_hook.so",
        "libascend_kernel_hook.so"
    };

    for (string &hookLib : hookLibNames) {
        Path hookLibPath = (Path(hookLibDir) / Path(hookLib)).Resolved();
        if (hookLibPath.Exists()) {
            hookLib = hookLibPath.ToString();
            LOG_INFO("Use preload lib [%s]", hookLib.c_str());
        } else {
            LOG_ERROR("No such preload lib [%s]", hookLibPath.ToString().c_str());
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

void Process::DoLaunch(const ExecCmd &cmd)
{
    // pass all env-vars from global variable "environ"
    execvpe(cmd.ExecPath().c_str(), cmd.ExecArgv(), environ);
    _exit(EXIT_FAILURE);
}

void Process::PostProcess(const ExecCmd &cmd)
{
    auto status = int32_t{};
    wait(&status);

    if (WIFEXITED(status)) {
        if (WEXITSTATUS(status) != EXIT_SUCCESS) {
            std::cout << "[msleaks] user program " << cmd.ExecPath() << " exited abnormally" << std::endl;
        }
    } else if (WIFSIGNALED(status)) {
        int sig = WTERMSIG(status);
        std::cout << "[msleaks] user program exited by signal: " << sig << std::endl;
    }

    return;
}

}  // namespace Leaks