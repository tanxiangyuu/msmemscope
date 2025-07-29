// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef FRAMEWORK_PROCESS_H
#define FRAMEWORK_PROCESS_H

#include <memory>
#include <string>
#include <map>
#include <set>
#include <thread>
#include "server_process.h"
#include "protocol.h"

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
    explicit Process(const Config &config);
    ~Process() = default;
    void Launch(const std::vector<std::string> &execParams);
private:
    void SetPreloadEnv();
    void DoLaunch(const ExecCmd &cmd);
    void PostProcess(const ExecCmd &cmd);

    void MsgHandle(size_t &clientId, std::string &msg);
    void RecordHandler(const ClientId &clientId, const RecordBuffer& record);
    void MemoryRecordPreprocess(const ClientId &clientId, const RecordBase &record);
private:
    std::unique_ptr<ServerProcess> server_;
    std::map<ClientId, Protocol> protocolList_;
    Config config_;
    std::set<RecordType> preprocessTypeList_ = {
        RecordType::MEMORY_RECORD,
        RecordType::ATB_MEMORY_POOL_RECORD,
        RecordType::PTA_CACHING_POOL_RECORD,
        RecordType::PTA_WORKSPACE_POOL_RECORD,
        RecordType::MEM_ACCESS_RECORD,
        RecordType::MINDSPORE_NPU_RECORD,
        RecordType::ADDR_INFO_RECORD,
    };
};

}

#endif