// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "client_process.h"

#include <map>
#include <thread>
#include <chrono>
#include <unistd.h>
#include "protocol.h"
#include "serializer.h"
#include "host_injection/core/FuncSelector.h"
#include "host_injection/utils/InjectLogger.h"

static Client* client_ = nullptr;
ClientProcess::ClientProcess(CommType type)
{
    client_ = new Client(type);
    while (!client_->Connect()) {
        // server 启动前 client 会连接失败，等待 100ms 后重试
        constexpr uint64_t connectRetryDuration = 100;
        std::this_thread::sleep_for(std::chrono::milliseconds(connectRetryDuration));
    }
}

ClientProcess &ClientProcess::GetInstance(CommType type)
{
    static ClientProcess instance(type);
    return instance;
}

ClientProcess::~ClientProcess()
{
}

void ClientProcess::Log(ClientLogLevel level, std::string msg)
{
    static std::map<ClientLogLevel, std::string> levelStrMap = {
        {ClientLogLevel::DEBUG, "[DEBUG]"},
        {ClientLogLevel::INFO, "[INFO] "},
        {ClientLogLevel::WARN, "[WARN] "},
        {ClientLogLevel::ERROR, "[ERROR]"}
    };

    std::string logMsg = levelStrMap[level] + " " + msg;
    Leaks::PacketHead head {Leaks::PacketType::LOG};
    std::string buffer = Leaks::Serialize<Leaks::PacketHead, uint64_t>(head, logMsg.size());
    buffer += logMsg;

    if (client_ != nullptr && Notify(buffer) < 0) {
        std::cout << "log report failed" << std::endl;
    }
    return;
}

// 通过工具侧配合wait实现阻塞功能
// 此处Notify的消息格式为[MESSAGE] xxx:xxx; 与log格式统一，降低工具侧消息解析复杂度
int ClientProcess::Notify(std::string const &msg)
{
    int32_t sentBytes = 0;
    std::size_t totalSent = 0;
    while (totalSent < msg.size()) {
        sentBytes = client_->Write(msg.substr(totalSent));
        // partial send return sent bytes
        if (sentBytes <= 0) {
            return totalSent;
        }
        totalSent += static_cast<std::size_t>(sentBytes);
    }
    return totalSent;
}

int ClientProcess::Wait(std::string& msg, uint32_t timeOut)
{
    std::string recvMsg;
    msg.clear();
    int len = 0;
    // 当msg.size()大于0时说明开始接收，当len不大于0时说明接收结束
    for (uint32_t count = 0; msg.size() == 0 || len > 0;) {
        // 底层为 SOCKET 实现时，socket read 本身存在超时时间为 1s，因此
        // 此接口中的 timeOut 也就是重试 read 的次数
        len = client_->Read(recvMsg);
        if (len > 0) {
            msg += recvMsg;
            continue;
        }
        // 读取成功不占用超时时间，读取失败则增加超时计数
        if (timeOut > 0 && count++ >= timeOut) {
            return static_cast<int>(msg.size());
        }
    }
    return static_cast<int>(msg.size());
}

int ClientProcess::TerminateWithSignal(int signal)
{
    return kill(getpid(), signal);
}

int ClientNotify(std::string msg)
{
    return ClientProcess::GetInstance().Notify(msg);
}

int ClientWait(std::string& msg)
{
    return ClientProcess::GetInstance().Wait(msg);
}

void ClientDebugLog(std::string msg)
{
    ClientProcess::GetInstance().Log(ClientLogLevel::DEBUG, msg);
}

void ClientInfoLog(std::string msg)
{
    ClientProcess::GetInstance().Log(ClientLogLevel::INFO, msg);
}

void ClientWarnLog(std::string msg)
{
    ClientProcess::GetInstance().Log(ClientLogLevel::WARN, msg);
}

void ClientErrorLog(std::string msg)
{
    ClientProcess::GetInstance().Log(ClientLogLevel::ERROR, msg);
}
