// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "server_process.h"
#include <thread>
#include "utility/log.h"

namespace Leaks {

ServerProcess::ServerProcess(CommType type)
{
    server_ = new Server(type);

    if (server_ == nullptr) {
        LOG_ERROR("Initial server failed");
        return;
    }
}

void ServerProcess::Start()
{
    if (server_ != nullptr) {
        server_->Start();
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
    return server_->Read(clientId, msg);
}

int ServerProcess::Notify(std::size_t clientId, const std::string& msg)
{
    return server_->Write(clientId, msg);
}

void ServerProcess::SetMsgHandlerHook(ClientMsgHandlerHook &&hook)
{
    if (server_ != nullptr) {
        server_->SetMsgHandlerHook(std::move(hook));
    }
}

void ServerProcess::SetClientConnectHook(ClientConnectHook &&hook)
{
    if (server_ != nullptr) {
        server_->SetClientConnectHook(std::move(hook));
    }
}

}
