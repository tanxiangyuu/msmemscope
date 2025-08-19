// Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

#include "domain_socket_server.h"
#include <iostream>
#include <thread>
#include <chrono>


namespace Leaks {

DomainSocketServer::DomainSocketServer():server_(new Server(CommType::SOCKET)) { }

bool DomainSocketServer::init()
{
    if (server_ != nullptr) {
        server_->Start();
        return true;
    } else {
        std::cout << "DomainSocketServer init faild" << std::endl;
        return false;
    }
}


bool DomainSocketServer::sent(std::size_t clientId, const std::string& msg, size_t& size)
{
    size = static_cast<size_t>(server_->Write(clientId, msg));
    if (size < msg.size()) {
        std::cout << "DomainSocketServer sent msg failed" << std::endl;
        return false;
    }
    return true;
}

bool DomainSocketServer::receive(std::size_t clientId, std::string& msg, size_t& size)
{
    // 如果设置了timeout，那么，等待一段时间后，需要释放
    size = static_cast<size_t>(server_->Read(clientId, msg));
    if (size < msg.size()) {
        std::cout << "DomainSocketServer receive msg failed" << std::endl;
        return false;
    }
    return true;
}

void DomainSocketServer::SetMsgHandlerHook(LeaksClientMsgHandlerHook &&hook)
{
    server_->SetMsgHandlerHook(std::move(hook));
}

void DomainSocketServer::SetClientConnectHook(LeaksClientConnectHook &&hook)
{
    server_->SetClientConnectHook(std::move(hook));
}

DomainSocketServer::~DomainSocketServer()
{
    delete server_;
}

}