// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "client_process.h"

#include <map>
#include <thread>
#include <chrono>
#include <unistd.h>
#include <cerrno>
#include "protocol.h"
#include "serializer.h"
#include "domain_socket_client.h"
#include "shared_memory_client.h"
#include "host_injection/core/FuncSelector.h"
#include "host_injection/utils/InjectLogger.h"

namespace Leaks {

ClientProcess::ClientProcess(LeaksCommType type) : logLevel_(Leaks::LogLv::INFO)
{
    char* test = std::getenv("LEAKS_TEST");
    if (test) {
        type = LeaksCommType::MEMORY_DEBUG;
    }
    if (LeaksCommType::DOMAIN_SOCKET == type) {
        client_ = new DomainSocketClient(CommType::SOCKET);
    } else if (LeaksCommType::SHARED_MEMORY == type) {
        client_ = new DomainSocketClient(CommType::SOCKET);
    } else if (LeaksCommType::MEMORY_DEBUG == type) {
        std::cout << "LEAKS_TEST" << std::endl;
        client_ = new DomainSocketClient(CommType::MEMORY);
    } else {
        client_ = nullptr; //  invalid type
    }
    if (client_ == nullptr) {
        std::cout << "Initial client failed" << std::endl;
        return;
    }
    client_->init();
}

ClientProcess &ClientProcess::GetInstance(LeaksCommType type)
{
    static ClientProcess instance(type);
    return instance;
}

ClientProcess::~ClientProcess()
{
}

void ClientProcess::Log(LogLv level, std::string msg, const std::string fileName, const uint32_t line)
{
    static std::map<LogLv, std::string> levelStrMap = {
        {LogLv::DEBUG, "[DEBUG]"},
        {LogLv::INFO, "[INFO] "},
        {LogLv::WARN, "[WARN] "},
        {LogLv::ERROR, "[ERROR]"}
    };

    if (level < logLevel_) {
        return ;
    }

    std::string logMsg = levelStrMap[level] + " [" + fileName + ":" + std::to_string(line) + "] " + msg;
    Leaks::PacketHead head {Leaks::PacketType::LOG, static_cast<size_t>(logMsg.size())};
    std::string buffer = Leaks::Serialize<Leaks::PacketHead>(head);
    buffer += logMsg;

    if (client_ != nullptr && Notify(buffer) < 0) {
        std::cout << "log report failed" << std::endl;
    }
    return;
}

void ClientProcess::SetLogLevel(LogLv level)
{
    logLevel_ = level;
}

// 通过工具侧配合wait实现阻塞功能
// 此处Notify的消息格式为[MESSAGE] xxx:xxx; 与log格式统一，降低工具侧消息解析复杂度
int ClientProcess::Notify(const std::string &msg)
{
    size_t sentBytes = msg.size();
    if (client_->sent(msg, sentBytes) == false) {
        std::cout << "client notify failed!" << std::endl;
        return 0;
    }
    return sentBytes;
}

int ClientProcess::Wait(std::string& msg, uint32_t timeOut)
{
    size_t len = 0;
    msg.clear();
    if (!client_->receive(msg, len, timeOut)) {
        std::cout << "client wait failed " << std::endl;
        return 0;
    }
    return len;
}

int ClientProcess::TerminateWithSignal(int signal)
{
    return kill(getpid(), signal);
}

}
