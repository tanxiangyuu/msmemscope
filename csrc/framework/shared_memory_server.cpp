// Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

#include "shared_memory_server.h"
#include <sys/mman.h>
#include <sys/statvfs.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include "utils.h"
#include "securec.h"


namespace Leaks {

SharedMemoryServer::SharedMemoryServer(): shmSize_(SHM_SIZE), c2sQueue_(nullptr), s2cBuffer_(nullptr), fdc2s_(-1),
    name_(nullptr), clientConnectHook_(nullptr), clientMsgHandlerHook_(nullptr), isRunning_(true) { }

uint64_t SharedMemoryServer::GetShmAvailable()
{
    struct statvfs buf;
    if (statvfs("/dev/shm", &buf) != 0) {
        std::cout << "statvfs /dev/shm failed" << std::endl;
        return 0;
    }
    return buf.f_bsize * buf.f_bavail;
}

bool SharedMemoryServer::Init()
{
    std::string name = "c2s_";
    name += Utility::GetDateStr();
    name_ = new char[name.size() + 1];
    strcpy_s(name_, name.size() + 1, name.c_str());

    fdc2s_ = shm_open(name_, O_CREAT | O_RDWR, 0600);
    shmSize_ = GetShmAvailable();
    if (fdc2s_ == -1 || shmSize_ == 0) {
        return false;
    }

    if (shmSize_ >= SHM_SIZE) {
        shmSize_ = SHM_SIZE;
    } else {
        return false;
    }

    if (ftruncate(fdc2s_, shmSize_) == -1) {
        return false;
    }

    void* ptr = mmap(nullptr, shmSize_, PROT_READ | PROT_WRITE, MAP_SHARED, fdc2s_, 0);
    if (ptr == MAP_FAILED) {
        std::cout << "server init failed" << std::endl;
        return false;
    }

    s2cBuffer_ = static_cast<uint8_t*>(ptr);

    c2sQueue_ = reinterpret_cast<Utility::LockFreeQueue*>(s2cBuffer_ + SHM_S2C_SIZE);
    c2sQueue_->ServerInit(shmSize_ - SHM_S2C_SIZE);

    if (clientConnectHook_) {
        clientConnectHook_(0);
    }

    receiveWorker_ = std::thread([this]() {
        while (isRunning_) {
            ClientId clientId = 0;
            size_t size = 0;
            std::string msg = "";
            bool flag = Receive(clientId, msg, size);
            if (!flag) {
                continue;
            }
            if (clientMsgHandlerHook_) {
                clientMsgHandlerHook_(clientId, msg);
            }
        }
    });

    if (setenv("SHM_NAME", name_, 1) == -1) {
        return false;
    }
    if (setenv("SHM_SIZE", std::to_string(shmSize_).c_str(), 1) == -1) {
        return false;
    }
    return true;
}

bool SharedMemoryServer::Send(std::size_t clientId, const std::string& msg, size_t& size)
{
    if (s2cBuffer_ == nullptr || size > SHM_S2C_SIZE) {
        return false;
    }
    new (s2cBuffer_) std::atomic<size_t>(0);
    size = msg.size();
    if (memcpy_s(s2cBuffer_ + sizeof(std::atomic<size_t>), shmSize_ - sizeof(std::atomic<size_t>), &size, sizeof(size_t))) {
        return false;
    }
    if (memcpy_s(s2cBuffer_ + sizeof(std::atomic<size_t>) + sizeof(size_t), shmSize_ - sizeof(std::atomic<size_t>) - sizeof(size_t), msg.data(), size)) {
        return false;
    }
    return true;
}

bool SharedMemoryServer::Receive(std::size_t& clientId, std::string& msg, size_t& size)
{
    if (c2sQueue_ == nullptr) {
        return false;
    }
    if (!c2sQueue_->DeQueue(msg, clientId)) {
        size = 0;
        return false;
    }
    size = msg.size();
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
    if (c2sQueue_ != nullptr) {
        while (!c2sQueue_->IsEmpty()) {}
    }
    isRunning_ = false;
    if (receiveWorker_.joinable()) {
        receiveWorker_.join();
    }
    if (s2cBuffer_ != nullptr) {
        munmap(s2cBuffer_, shmSize_);
    }
    if (fdc2s_ != -1) {
        close(fdc2s_);
    }
    if (name_ != nullptr) {
        shm_unlink(name_);
    }
}

}