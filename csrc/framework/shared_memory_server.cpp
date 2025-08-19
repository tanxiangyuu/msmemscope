// Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

#include "shared_memory_server.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include "utils.h"


namespace Leaks {

SharedMemoryServer::SharedMemoryServer(): c2sQueue_(nullptr), s2cBuffer_(nullptr), fd_c2s_(-1), name_(nullptr), 
    clientConnectHook_(nullptr), clientMsgHandlerHook_(nullptr), isRunning_(true) { }

bool SharedMemoryServer::init()
{
    std::string name = "c2s_";
    name += Utility::GetDateStr();
    name_ = new char[name.size() + 1];
    strcpy(name_, name.c_str());

    fd_c2s_ = shm_open(name_, O_CREAT | O_RDWR, 0666);
    if (ftruncate(fd_c2s_, SHM_SIZE) == -1) {
        return false;
    }

    s2cBuffer_ = static_cast<uint8_t*>(mmap(nullptr, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd_c2s_, 0));
    c2sQueue_ = new Utility::LockFreeQueue(SHM_SIZE, s2cBuffer_ + SHM_S2C_SIZE);

    if (clientConnectHook_) {
        clientConnectHook_(0);
    }

    receiveWorker_ = std::thread([this]() {
        while (isRunning_) {
            ClientId clientId = 0;
            size_t size;
            std::string msg;
            bool flag = receive(clientId, msg, size);
            if (!flag) {
                continue;
            }
            clientMsgHandlerHook_(clientId, msg);
        }
    });

    if (setenv("SHM_NAME", name_, 1) == -1) {
        return false;
    }
    return true;
}

bool SharedMemoryServer::sent(std::size_t clientId, const std::string& msg, size_t& size)
{
    if (s2cBuffer_ == nullptr || size > SHM_S2C_SIZE) {
        return false;
    }
    std::memcpy_s(s2cBuffer_, msg.data(), msg.size());
    return true;
}

bool SharedMemoryServer::receive(std::size_t clientId, std::string& msg, size_t& size)
{
    char* data{};
    if (!c2sQueue_->dequeue(data, size)) {
        return false;
    }
    msg = std::string(data, size);
    return true;
}

void SharedMemoryServer::SetMsgHandlerHook(LeaksClientMsgHandlerHook &&hook)
{
    clientMsgHandlerHook_ = hook;
}

void SharedMemoryServer::SetClientConnectHook(LeaksClientConnectHook &&hook)
{
    clientConnectHook_ = hook;
}

SharedMemoryServer::~SharedMemoryServer()
{
    isRunning_ = false;
    if (receiveWorker_.joinable()) {
        receiveWorker_.join();
    }
    munmap(s2cBuffer_, SHM_SIZE);
    close(fd_c2s_);
    shm_unlink(name_);
}

}