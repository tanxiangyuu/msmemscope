// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "client_process.h"

#include <map>
#include <thread>
#include <chrono>
#include <unistd.h>
#include <cerrno>
#include "protocol.h"
#include "serializer.h"
#include "host_injection/core/FuncSelector.h"
#include "host_injection/utils/InjectLogger.h"

namespace Leaks {

constexpr uint32_t MAX_TRY_COUNT = 100;

static Client* client_ = nullptr;
ClientProcess::ClientProcess(CommType type)
{
    client_ = new Client(type);

    if (client_ == nullptr) {
        std::cout << "Initial client failed" << std::endl;
        return;
    }
    uint32_t count = 0;
    while (!client_->Connect() && count < MAX_TRY_COUNT) {
        // server 启动前 client 会连接失败，等待 100ms 后重试
        constexpr uint64_t connectRetryDuration = 100;
        count++;
        std::this_thread::sleep_for(std::chrono::milliseconds(connectRetryDuration));
    }
    if (count == MAX_TRY_COUNT) {
        std::cout << "connect to server fail" << std::endl;
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
    std::lock_guard<std::mutex> lock(notifyMutex_);
    int32_t sentBytes = 0;
    std::size_t totalSent = 0;
    /* 循环外部先发送一次，减少重复构造字符串 */
    totalSent = client_->Write(msg);
    while (totalSent < msg.size()) {
        sentBytes = client_->Write(msg.substr(totalSent));
        // partial send return sent bytes
        if (sentBytes <= 0) {
            if (errno == EPIPE || errno == EBADF || errno == ENOTCONN || errno == ECONNRESET) {
                return -1;
            }
            return totalSent;
        }
        totalSent += static_cast<std::size_t>(sentBytes);
    }
    return totalSent;
}

int ClientProcess::Wait(std::string& msg, uint32_t timeOut)
{
    if (client_ == nullptr) {
        std::cout << "Client doesn't exist! ClientProcess wait failed!" << std::endl;
        return static_cast<int>(msg.size());
    }
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
        if (count >= timeOut) {
            return static_cast<int>(msg.size());
        }
        count++;
    }
    return static_cast<int>(msg.size());
}

int ClientProcess::TerminateWithSignal(int signal)
{
    return kill(getpid(), signal);
}

}
