// Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

#ifndef DOMAIN_SOCKET_CLIENT_H
#define DOMAIN_SOCKET_CLIENT_H

#include "communication_proxy_client.h"
#include "host_injection/core/Communication.h"

namespace Leaks{

class DomainSocketClient : public CommunicationProxyClient{
public:
    explicit DomainSocketClient();
    ~DomainSocketClient() override;
    bool init() override;
    bool sent(const std::string& msg, size_t& size) override;
    bool receive(std::string& msg, size_t& size, uint32_t timeOut) override;
private:
    Client* client_;
};

}

#endif