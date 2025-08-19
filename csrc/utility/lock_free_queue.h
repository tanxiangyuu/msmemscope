// Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

#ifndef LOCK_FREE_QUEUE_H
#define LOCK_FREE_QUEUE_H

#include <atomic>
#include <cstddef>

namespace Utility {

struct DataHeader {
    uint32_t magic_prefix;
    size_t data_size;
    explicit DataHeader(size_t size):magic_prefix(0xABCD1234), data_size(size) {}
};

class LockFreeQueue {
public:
    explicit LockFreeQueue(size_t total_memory_size, uint8_t* buffer);
    bool enqueue(const void* data, size_t data_size);
    bool dequeue(void* output, size_t& out_size);
private:
    size_t getNextData();
    const size_t total_memory_size_;
    uint8_t* buffer_;
    size_t head_;
    size_t tail_;
    size_t free_space_;
    std::atomic<uint8_t> flag_;
};

}

#endif