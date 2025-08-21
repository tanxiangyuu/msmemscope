// Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

#ifndef DOMAIN_SOCKET_SERVER_H
#define DOMAIN_SOCKET_SERVER_H

#include "communication_proxy_server.h"
#include "host_injection/core/Communication.h"

namespace Leaks {

class DomainSocketServer : public CommunicationProxyServer {
public:
    explicit DomainSocketServer(CommType type);
    ~DomainSocketServer() override;
    bool init() override;
    bool send(std::size_t clientId, const std::string& msg, size_t& size) override;
    bool receive(std::size_t& clientId, std::string& msg, size_t& size) override;
    void SetMsgHandlerHook(LeaksClientMsgHandlerHook &&hook) override;
    void SetClientConnectHook(LeaksClientConnectHook &&hook) override;
private:
    Server* server_;
};

}

#endif