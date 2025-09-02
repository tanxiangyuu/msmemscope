// Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

#ifndef COMMUNICATION_PROXY_CLIENT_H
#define COMMUNICATION_PROXY_CLIENT_H

#include <string>
#include <cstddef>
#include "comm_def.h"

namespace Leaks {

class CommunicationProxyClient {
public:
    CommunicationProxyClient() {}
    virtual bool Init() = 0;
    virtual bool Send(const std::string& msg, size_t& size) = 0;
    virtual bool Receive(std::string& msg, size_t& size, uint32_t timeOut) = 0;
    virtual ~CommunicationProxyClient() = default;
};

}

#endif