// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef __CORE_CLIENT_PROCESS_H__
#define __CORE_CLIENT_PROCESS_H__

#include <string>
#include <csignal>
#include "host_injection/core/Communication.h"
#include "log.h"

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
    void Log(ClientLogLevel level, std::string msg, const std::string fileName, const uint32_t line);
    int Notify(std::string const &msg);
    int Wait(std::string& msg, uint32_t timeOut = 10);
    int TerminateWithSignal(int signal = SIGINT);
};

int ClientNotify(std::string msg);

int ClientWait(std::string& msg);

#define CLIENT_DEBUG_LOG(format)                                                                                      \
    do {                                                                                                              \
        Leaks::ClientProcess::GetInstance().Log(Leaks::ClientLogLevel::DEBUG, format,                                 \
            Utility::GetLogSourceFileName(__FILE__),                                                                  \
            __LINE__);                                                                                                \
    } while (0)

#define CLIENT_INFO_LOG(format)                                                                                       \
    do {                                                                                                              \
        Leaks::ClientProcess::GetInstance().Log(Leaks::ClientLogLevel::INFO,                                          \
            format, Utility::GetLogSourceFileName(__FILE__),                                                          \
            __LINE__);                                                                                                \
    } while (0)

#define CLIENT_WARN_LOG(format)                                                                                       \
    do {                                                                                                              \
        Leaks::ClientProcess::GetInstance().Log(Leaks::ClientLogLevel::INFO,                                          \
            format, Utility::GetLogSourceFileName(__FILE__),                                                          \
            __LINE__);                                                                                                \
    } while (0)

#define CLIENT_ERROR_LOG(format)                                                                                      \
    do {                                                                                                              \
        Leaks::ClientProcess::GetInstance().Log(Leaks::ClientLogLevel::INFO,                                          \
            format, Utility::GetLogSourceFileName(__FILE__),                                                          \
            __LINE__);                                                                                                \
    } while (0)

}

#endif // __CORE_CLIENT_PROCESS_H__
