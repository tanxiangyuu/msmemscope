// Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

#include "shared_memory_server.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include "utils.h"
#include "securec.h"


namespace Leaks {

SharedMemoryServer::SharedMemoryServer(): c2sQueue_(nullptr), s2cBuffer_(nullptr), fd_c2s_(-1), name_(nullptr),
    clientConnectHook_(nullptr), clientMsgHandlerHook_(nullptr), isRunning_(true) { }

bool SharedMemoryServer::init()
{
    std::string name = "c2s_";
    name += Utility::GetDateStr();
    name_ = new char[name.size() + 1];
    strcpy_s(name_, name.size() + 1, name.c_str());

    // 1. 动态加载 librt.so.1
    void *handle = dlopen("librt.so.1", RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    // 2. 获取 shm_open 函数指针
    int (*shm_open_ptr)(const char *, int, mode_t);
    shm_open_ptr = (int (*)(const char *, int, mode_t))dlsym(handle, "shm_open");
    if (!shm_open_ptr) {
        fprintf(stderr, "dlsym failed: %s\n", dlerror());
        dlclose(handle);
        return 1;
    }

    fd_c2s_ = shm_open_ptr(name_, O_CREAT | O_RDWR, 0666);
    if (ftruncate(fd_c2s_, SHM_SIZE) == -1) {
        return false;
    }

    void* ptr = mmap(nullptr, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd_c2s_, 0);
    if (ptr == MAP_FAILED) {
        std::cout << "server init failed" << std::endl;
    }

    if (memset_s(ptr, SHM_SIZE, 0, SHM_SIZE) != EOK) {
        return false;
    }

    s2cBuffer_ = static_cast<uint8_t*>(ptr);

    c2sQueue_ = new Utility::LockFreeQueue(SHM_SIZE - SHM_S2C_SIZE - 1 - 2 * sizeof(size_t), s2cBuffer_ + SHM_S2C_SIZE);

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
            if (clientMsgHandlerHook_) {
                clientMsgHandlerHook_(clientId, msg);
            }
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
    size = msg.size();
    if (memcpy_s(s2cBuffer_ + sizeof(size_t), SHM_SIZE - sizeof(size_t), &size, sizeof(size_t))) {
        return false;
    }
    if (memcpy_s(s2cBuffer_ + sizeof(size_t) + sizeof(size_t), SHM_SIZE - sizeof(size_t) - sizeof(size_t), msg.data(), size)) {
        return false;
    }
    return true;
}

bool SharedMemoryServer::receive(std::size_t& clientId, std::string& msg, size_t& size)
{
    std::lock_guard<std::mutex> lock(sentMutex_);
    char* data{};
    if (!c2sQueue_->dequeue((void**)&data, size, clientId)) {
        return false;
    }
    std::cout << "server receive success" << std::endl;
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
    
    void *handle = dlopen("librt.so.1", RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    int (*shm_unlink_ptr)(const char *);
    shm_unlink_ptr = (int (*)(const char *))dlsym(handle, "shm_unlink")
    if (!shm_unlink_ptr) {
        fprintf(stderr, "dlsym failed: %s\n", dlerror());
        dlclose(handle);
        return 1;
    }

    shm_unlink_ptr(name_);
}

}