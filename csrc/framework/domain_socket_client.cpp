// Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

#include <iostream>
#include <thread>
#include <chrono>
#include "domain_socket_client.h"

namespace Leaks {

constexpr uint32_t MAX_TRY_COUNT = 100;

DomainSocketClient::DomainSocketClient():client_(new Client(CommType::SOCKET)) { }

bool DomainSocketClient::init()
{
    if (client_ == nullptr) {
        std::cout << "Initial client failed" << std::endl;
        return false;
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
        return false;
    }
    return true;
}

bool DomainSocketClient::sent(const std::string& msg, size_t& size)
{
    int32_t sentBytes = 0;
    /* 循环外部先发送一次，减少重复构造字符串 */
    size = static_cast<size_t>(client_->Write(msg));
    while (size < msg.size()) {
        sentBytes = client_->Write(msg.substr(size));
        // partial send return sent bytes
        if (sentBytes <= 0) {
            return false;
        }
        size += static_cast<std::size_t>(sentBytes);
    }
    return true;
}

bool DomainSocketClient::receive(std::string& msg, size_t& size, uint32_t timeOut)
{
    std::string recvMsg;
    int len = 0;
    // 当msg.size()大于0时说明开始接收，当len不大于0时说明接收结束
    for (uint32_t count = 0; msg.size() == 0 || len > 0;) {
        // 底层为 SOCKET 实现时，socket read 本身存在超时时间为 1s，因此
        // 此接口中的 timeOut 也就是重试 read 的次数
        len = client_->Read(recvMsg);
        if (len > 0) {
            msg += recvMsg;
            size += static_cast<size_t>(len);
            continue;
        }
        if (count >= timeOut) {
            return false;
        }
    }
    return true;
}

DomainSocketClient::~DomainSocketClient()
{
    delete client_;
}

}