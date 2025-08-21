// Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

#ifndef SHARED_MEMORY_CLIENT_H
#define SHARED_MEMORY_CLIENT_H

#include "communication_proxy_client.h"
#include "lock_free_queue.h"

namespace Leaks {

class SharedMemoryClient : public CommunicationProxyClient {
public:
    explicit SharedMemoryClient();
    ~SharedMemoryClient() override;
    bool init() override;
    bool sent(const std::string& msg, size_t& size) override;
    bool receive(std::string& msg, size_t& size, uint32_t timeOut) override;
private:
    Utility::LockFreeQueue* c2sQueue_;
    uint8_t* s2cBuffer_;
    int fd_c2s_;
    char* name_;
    ClientId clientId_;
};

}

#endif