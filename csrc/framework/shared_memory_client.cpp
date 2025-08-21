// Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

#include "shared_memory_client.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <iostream>
#include <cstring>
#include "securec.h"

namespace Leaks {

SharedMemoryClient::SharedMemoryClient() : c2sQueue_(nullptr), s2cBuffer_(nullptr), fd_c2s_(-1), name_(nullptr),
    clientId_(0){ }

bool SharedMemoryClient::init()
{
    name_ = std::getenv("SHM_NAME");
    if (!name_ || std::string(name_).empty()) {
        std::cout << "[msleaks] Failed to acquire SHM_NAME environment variable while SharedMemoryClient init Failed" << std::endl;
        return false;
    }

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
    fd_c2s_ = shm_open_ptr(name_, O_RDWR, 0666);
    if (fd_c2s_ == -1) {
        std::cout << "[msleaks] Failed to open shared memory. SharedMemoryClient init Failed.\n";
        return false;
    }

    s2cBuffer_ = static_cast<uint8_t*>(mmap(nullptr, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd_c2s_, 0));
    if (s2cBuffer_ == nullptr) {
        std::cout << "[msleaks] Failed to map shared memory. SharedMemoryClient init Failed.\n";
        return false;
    }

    c2sQueue_ = new Utility::LockFreeQueue(SHM_SIZE - SHM_S2C_SIZE - 1 - 2 * sizeof(size_t), s2cBuffer_ + SHM_S2C_SIZE);
    return true;
}

bool SharedMemoryClient::sent(const std::string& msg, size_t& size)
{
    return c2sQueue_->enqueue((const void*) msg.data(), size, clientId_);
}

bool SharedMemoryClient::receive(std::string& msg, size_t& size, uint32_t timeOut)
{
    if (s2cBuffer_ == nullptr) {
        return false;
    }

    if (memcpy_s(&clientId_, sizeof(size_t), s2cBuffer_, sizeof(size_t))) {
        return false;
    }

    std::atomic<ClientId>* atomicPtr = reinterpret_cast<std::atomic<ClientId>*>(s2cBuffer_);

    while (!atomicPtr->compare_exchange_weak(clientId_, clientId_+1)) { }

    if (memcpy_s(&size, sizeof(size_t), s2cBuffer_ + sizeof(size_t), sizeof(size_t))) {
        return false;
    }
    uint8_t* ptr = new uint8_t[size];
    if (memcpy_s(ptr, size, s2cBuffer_ + sizeof(size_t) + sizeof(size_t), size)) {
        return false;
    }
    msg = std::string(reinterpret_cast<char*>(ptr), size);
    return true;
}

SharedMemoryClient::~SharedMemoryClient() {}

}