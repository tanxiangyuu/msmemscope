// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "server_process.h"
#include <thread>
#include "utility/log.h"
#include "domain_socket_server.h"
#include "shared_memory_server.h"

namespace Leaks {

ServerProcess::ServerProcess(LeaksCommType type)
{
    if (LeaksCommType::DOMAIN_SOCKET == type) {
        server_ = new DomainSocketServer(CommType::SOCKET);
    } else if (LeaksCommType::SHARED_MEMORY == type) {
        server_ = new DomainSocketServer(CommType::SOCKET);
    } else if (LeaksCommType::MEMORY_DEBUG == type) {
        server_ = new DomainSocketServer(CommType::MEMORY);
    } else {
        server_ = nullptr; //  invalid type
    }
    if (server_ == nullptr) {
        std::cout << "ServerProcess initial server failed" << std::endl;
        return;
    }
}

void ServerProcess::Start()
{
    if (server_ != nullptr) {
        server_->init();
    }
}

ServerProcess::~ServerProcess()
{
    if (server_ != nullptr) {
        delete server_;
        server_ = nullptr;
    }
}

int ServerProcess::Wait(std::size_t clientId, std::string& msg)
{
    // 如果设置了timeout，那么，等待一段时间后，需要释放
    size_t size = 0;
    if (server_ != nullptr) {
        server_->receive(clientId, msg, size);
    }
    return size;
}

int ServerProcess::Notify(std::size_t clientId, const std::string& msg)
{
    size_t size = 0;
    if (server_ != nullptr) {
        server_->sent(clientId, msg, size);
    }
    return size;
}

void ServerProcess::SetMsgHandlerHook(LeaksClientMsgHandlerHook &&hook)
{
    if (server_ != nullptr) {
        server_->SetMsgHandlerHook(std::move(hook));
    }
}

void ServerProcess::SetClientConnectHook(LeaksClientConnectHook &&hook)
{
    if (server_ != nullptr) {
        server_->SetClientConnectHook(std::move(hook));
    }
}

}
