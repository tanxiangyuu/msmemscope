// Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

#include "shared_memory_client.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <iostream>
#include <cstring>

namespace Leaks {

SharedMemoryClient::SharedMemoryClient() : c2sQueue_(nullptr), s2cBuffer_(nullptr), fd_c2s_(-1), name_(nullptr){ }

bool SharedMemoryClient::init()
{
    name_ = std::getenv("SHM_NAME");
    if (!name_ || std::string(name_).empty()) {
        std::cout << "[msleaks] Failed to acquire SHM_NAME environment variable while SharedMemoryClient init Failed" << std::endl;
        return false;
    }
    fd_c2s_ = shm_open(name_, O_RDWR, 0666);
    if (fd_c2s_ == -1) {
        std::cout << "[msleaks] Failed to open shared memory. SharedMemoryClient init Failed.\n";
        return false;
    }
    s2cBuffer_ = static_cast<uint8_t*>(mmap(nullptr, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd_c2s_, 0));

    c2sQueue_ = new Utility::LockFreeQueue(SHM_SIZE, s2cBuffer_ + SHM_S2C_SIZE);
    return true;
}

bool SharedMemoryClient::sent(const std::string& msg, size_t& size)
{
    return c2sQueue_->enqueue((void*) &msg, size);
}

bool SharedMemoryClient::receive(std::string& msg, size_t& size, uint32_t timeOut)
{
    if (s2cBuffer_ == nullptr) {
        return false;
    }

    std::memcpy_s(&size, s2cBuffer_, sizeof(size_t));
    char* data{};
    std::memcpy_s(data, s2cBuffer_ + sizeof(size_t), size);
    msg = std::string(data);
    return true;
}

SharedMemoryClient::~SharedMemoryClient() {}

}