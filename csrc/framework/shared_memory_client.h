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
    bool Init() override;
    bool Send(const std::string& msg, size_t& size) override;
    bool Receive(std::string& msg, size_t& size, uint32_t timeOut) override;
private:
    uint64_t shmSize_;
    Utility::LockFreeQueue* c2sQueue_;
    uint8_t* s2cBuffer_;
    int fdc2s_;
    std::string name_;
    ClientId clientId_;
};

}

#endif