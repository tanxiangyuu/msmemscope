// Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

#ifndef COMMUNICATION_PROXY_SERVER_H
#define COMMUNICATION_PROXY_SERVER_H

#include <string>
#include <cstddef>
#include <functional>
#include "comm_def.h"

namespace Leaks {

using LeaksClientConnectHook = std::function<void(ClientId)>;
using LeaksClientMsgHandlerHook = std::function<void(ClientId&, std::string&)>;

class CommunicationProxyServer {
public:
    CommunicationProxyServer() {}
    virtual bool Init() = 0;
    virtual bool Send(std::size_t clientId, const std::string& msg, size_t& size) = 0;
    virtual bool Receive(std::size_t& clientId, std::string& msg, size_t& size) = 0;
    virtual void SetMsgHandlerHook(LeaksClientMsgHandlerHook &&hook) = 0;
    virtual void SetClientConnectHook(LeaksClientConnectHook &&hook) = 0;
    virtual ~CommunicationProxyServer() = default;
};

}

#endif