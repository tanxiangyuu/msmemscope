// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef FRAMEWORK_PROCESS_H
#define FRAMEWORK_PROCESS_H

#include <memory>
#include <string>
#include <map>
#include <set>
#include <thread>
#include "event.h"
#include "comm_def.h"

namespace MemScope {
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
    static Process& GetInstance(Config config);
    explicit Process(const Config &config);
    void Launch(const std::vector<std::string> &execParams);
    void RecordHandler(const RecordBuffer& record);
private:
    void SetPreloadEnv();
    void DoLaunch(const ExecCmd &cmd);

    std::shared_ptr<EventBase> RecordToEvent(RecordBase* record);
    void MemoryRecordPreprocess(const ClientId &clientId, const RecordBase &record);
private:
    Config config_;
};

void EventHandler(std::shared_ptr<EventBase> event);

}

#endif