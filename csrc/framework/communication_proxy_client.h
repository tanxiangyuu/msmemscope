// Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

#ifndef COMMUNICATION_PROXY_CLIENT_H
#define COMMUNICATION_PROXY_CLIENT_H

#include <string>
#include <cstddef>

namespace Leaks {

enum class LeaksCommType {
    SHARED_MEMORY,
    DOMAIN_SOCKET
};

class CommunicationProxyClient {
public:
    CommunicationProxyClient() {}
    virtual bool init() = 0;
    virtual bool sent(const std::string& msg, size_t& size) = 0;
    virtual bool receive(std::string& msg, size_t& size, uint32_t timeOut) = 0;
    virtual ~CommunicationProxyClient() = default;
};

}

#endif