// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef LOCK_FREE_QUEUE_H
#define LOCK_FREE_QUEUE_H

#include <string>
#include <atomic>
#include <cstddef>
#include "securec.h"

namespace Utility {

class LockFreeQueue {
public:
    bool ServerInit(uint64_t size);
    bool ClientInit();
    bool EnQueue(const void* data, size_t data_size, size_t id);
    bool DeQueue(std::string& msg, size_t& id);
    bool IsEmpty() const {return head_.load(std::memory_order_relaxed) == tail_.load(std::memory_order_relaxed);}

private:
    uint32_t magic_;
    uint64_t size_;
    /* uint64_t的长度可以表达EB级的数据量，实际工程中完全够用，不会产生回绕 */
    std::atomic<uint64_t> head_;
    std::atomic<uint64_t> tail_;

    LockFreeQueue() = delete;
    ~LockFreeQueue() = delete;
    LockFreeQueue(const LockFreeQueue&) = delete;
    LockFreeQueue& operator=(const LockFreeQueue&) = delete;
    LockFreeQueue(LockFreeQueue&& other) = delete;
    LockFreeQueue& operator=(LockFreeQueue&& other) = delete;
};

} // namespace Utility

#endif