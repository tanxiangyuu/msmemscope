// Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

#ifndef LOCK_FREE_QUEUE_H
#define LOCK_FREE_QUEUE_H

#include <atomic>
#include <cstddef>

namespace Utility {

constexpr size_t OUTPUT_BUF_SIZE = 1 * 1024 * 1024;

struct DataHeader {
    uint32_t magic_prefix;
    size_t data_size;
    size_t clientId;
    explicit DataHeader(size_t size, size_t id):magic_prefix(0xABCD1234), data_size(size), clientId(id) {}
};

class LockFreeQueue {
public:
    explicit LockFreeQueue(size_t total_memory_size, uint8_t* buffer);
    bool enqueue(const void* data, size_t data_size, size_t id);
    bool dequeue(void** output, size_t& out_size, size_t& id);
private:
    size_t getNextData();
    const size_t total_memory_size_;
    uint8_t* buffer_;
    size_t head_;
    size_t tail_;
    size_t free_space_;
    std::atomic<uint8_t>* flag_;
    DataHeader* dataHeader_;
    uint8_t* outputBuf_;
};

}

#endif