// Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

#ifndef SHARED_MEMORY_SERVER_H
#define SHARED_MEMORY_SERVER_H
#include <mutex>
#include <thread>
#include "communication_proxy_server.h"
#include "lock_free_queue.h"

namespace Leaks {

class SharedMemoryServer : public CommunicationProxyServer {
public:
    explicit SharedMemoryServer();
    ~SharedMemoryServer() override;
    bool init() override;
    bool send(std::size_t clientId, const std::string& msg, size_t& size) override;
    bool receive(std::size_t& clientId, std::string& msg, size_t& size) override;
    void SetMsgHandlerHook(LeaksClientMsgHandlerHook &&hook) override;
    void SetClientConnectHook(LeaksClientConnectHook &&hook) override;
private:
    std::mutex sentMutex_;
    Utility::LockFreeQueue* c2sQueue_;
    uint8_t* s2cBuffer_;
    int fd_c2s_;
    char* name_;
    LeaksClientConnectHook clientConnectHook_;
    LeaksClientMsgHandlerHook clientMsgHandlerHook_;
    bool isRunning_;
    std::thread receiveWorker_;
};

}

#endif