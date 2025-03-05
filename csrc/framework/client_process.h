// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef __CORE_CLIENT_PROCESS_H__
#define __CORE_CLIENT_PROCESS_H__

#include <string>
#include <csignal>
#include "host_injection/core/Communication.h"

namespace Leaks {

enum class ClientLogLevel : uint8_t {
    DEBUG = 0U,
    INFO,
    WARN,
    ERROR,
};

class ClientProcess {
public:
    explicit ClientProcess(CommType type);
    static ClientProcess &GetInstance(CommType type = CommType::SOCKET);
    ~ClientProcess();
    void Log(ClientLogLevel level, std::string msg);
    int Notify(std::string const &msg);
    int Wait(std::string& msg, uint32_t timeOut = 10);
    int TerminateWithSignal(int signal = SIGINT);
};

int ClientNotify(std::string msg);

int ClientWait(std::string& msg);

void ClientDebugLog(std::string msg);

void ClientInfoLog(std::string msg);

void ClientWarnLog(std::string msg);

void ClientErrorLog(std::string msg);

}

#endif // __CORE_CLIENT_PROCESS_H__
