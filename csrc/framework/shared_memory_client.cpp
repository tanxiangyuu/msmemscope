// Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

#include "shared_memory_client.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <iostream>
#include <unistd.h>
#include <cstring>
#include "securec.h"

namespace Leaks {

SharedMemoryClient::SharedMemoryClient() : shmSize_(SHM_SIZE), c2sQueue_(nullptr),
    s2cBuffer_(nullptr), fdc2s_(-1), name_(""), clientId_(0){ }

bool GetUint64Env(const char* name, uint64_t& outValue)
{
    const char* strValue = std::getenv(name);
    if (strValue == nullptr) {
        std::cerr << "[ERROR] env " << name << " not found" << std::endl;
        return false;
    }

    // 将字符串转换为 uint64_t
    char* endPtr = nullptr;
    errno = 0; // 清除错误状态
    uint64_t value = std::strtoull(strValue, &endPtr, 10); // 10 表示十进制

    // 检查转换是否成功
    if (errno != 0 || *endPtr != '\0') {
        std::cerr << "[ERROR] can't convert env " << name << " to uint64_t " << strerror(errno) << std::endl;
        return false;
    }

    outValue = value;
    return true;
}

bool SharedMemoryClient::Init()
{
    if (const char* envname = std::getenv("SHM_NAME")) {
        name_ = envname;
    } else {
        std::cout << "[msleaks] Failed to acquire SHM_NAME environment variable while SharedMemoryClient init Failed" << std::endl;
        return false;
    }

    fdc2s_ = shm_open(name_.c_str(), O_RDWR, 0600);

    if (!GetUint64Env("SHM_SIZE", shmSize_)) {
        std::cout << "[msleaks] Failed to acquire SHM_SIZE environment variable while SharedMemoryClient init Failed" << std::endl;
        return false;
    }

    if (fdc2s_ == -1) {
        std::cout << "[msleaks] Failed to open shared memory. SharedMemoryClient init Failed.\n";
        return false;
    }

    void* ptr = mmap(nullptr, shmSize_, PROT_READ | PROT_WRITE, MAP_SHARED, fdc2s_, 0);
    if (ptr == MAP_FAILED) {
        std::cout << "[msleaks] Failed to map shared memory. SharedMemoryClient init Failed." << std::endl;
        return false;
    }

    s2cBuffer_ = static_cast<uint8_t*>(ptr);

    c2sQueue_ = reinterpret_cast<Utility::LockFreeQueue*>(s2cBuffer_ + SHM_S2C_SIZE);
    c2sQueue_->ClientInit();
    return true;
}

bool SharedMemoryClient::Send(const std::string& msg, size_t& size)
{
    return c2sQueue_->EnQueue((const void*) msg.data(), size, clientId_);
}

bool SharedMemoryClient::Receive(std::string& msg, size_t& size, uint32_t timeOut)
{
    if (s2cBuffer_ == nullptr) {
        return false;
    }

    std::atomic<ClientId>* atomicPtr = reinterpret_cast<std::atomic<ClientId>*>(s2cBuffer_);

    clientId_ = atomicPtr->fetch_add(1, std::memory_order_relaxed);

    if (memcpy_s(&size, sizeof(size_t), s2cBuffer_ + sizeof(std::atomic<ClientId>), sizeof(size_t))) {
        return false;
    }
    uint8_t* ptr = new uint8_t[size];
    if (memcpy_s(ptr, size, s2cBuffer_ + sizeof(std::atomic<ClientId>) + sizeof(size_t), size)) {
        delete[] ptr;
        return false;
    }
    msg = std::string(reinterpret_cast<char*>(ptr), size);
    delete[] ptr;
    return true;
}

SharedMemoryClient::~SharedMemoryClient()
{
    if (s2cBuffer_ != nullptr) {
        munmap(s2cBuffer_, shmSize_);
        s2cBuffer_ = nullptr;
    }
    if (fdc2s_ != -1) {
        close(fdc2s_);
        fdc2s_ = -1;
    }
}

}