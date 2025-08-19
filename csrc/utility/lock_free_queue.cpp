// Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

#include "lock_free_queue.h"
#include <cstring>
#include <iostream>
#include <cstddef>


namespace Utility {

LockFreeQueue::LockFreeQueue(size_t total_memory_size, uint8_t* buffer):total_memory_size_(total_memory_size), 
    buffer_(buffer), head_(0), tail_(0), free_space_(0) { }

bool LockFreeQueue::enqueue(const void* data, size_t data_size)
{
    flag_ = *buffer_;
    uint8_t old = 1;
    while (!flag_.compare_exchange_weak(old, 0)) {
        old = 1;
    }
    std::memcpy_s(&head_, buffer_ + 1, sizeof(size_t));
    std::memcpy_s(&tail_, buffer_ + 5, sizeof(size_t));
    
    free_space_ = tail_ >= head_ ? total_memory_size_ - (tail_ - head_) : head_ - tail_;

    size_t required_size = data_size + sizeof(DataHeader);
    
    if (free_space_ < required_size) {
        std::cout << "LockFreeQueue don't have enough space";
        old = 0;
        while (!flag_.compare_exchange_weak(old, 1)) {
            old = 0;
        }
        return false;
    }

    bool needs_truncation = tail_ + data_size > total_memory_size_;

    size_t old_tail = tail_;

    tail_ = needs_truncation ? required_size - (total_memory_size_ - tail_) : tail_ + required_size;
    std::memcpy_s(buffer_ + 4, &tail_, sizeof(size_t));
    old = 0;
    while (!flag_.compare_exchange_weak(old, 1)) {
        old = 0;
    }

    DataHeader dataHeader(required_size);

    if (needs_truncation) {
        size_t tail_size = total_memory_size_ - old_tail;
        if (tail_size < sizeof(DataHeader)) {
            std::memcpy_s(buffer_ + 9 + (sizeof(DataHeader) - tail_size), data, data_size);
            std::memcpy_s(buffer_ + 9 + old_tail, (uint8_t*)&dataHeader, tail_size);
            std::memcpy_s(buffer_ + 9, ((uint8_t*)&dataHeader) + tail_size, sizeof(DataHeader) - tail_size);
        } else {
            std::memcpy_s(buffer_ + 9 + old_tail + sizeof(DataHeader), data, tail_size - sizeof(DataHeader));
            std::memcpy_s(buffer_ + 9, (uint8_t*)data + (tail_size - sizeof(DataHeader)), data_size - (tail_size - sizeof(DataHeader)));
            std::memcpy_s(buffer_ + 9 + old_tail, (uint8_t*)&dataHeader, sizeof(DataHeader));
        }
    } else {
        std::memcpy_s(buffer_ + 9 + old_tail + sizeof(DataHeader), data, data_size);
        std::memcpy_s(buffer_ + 9 + old_tail, (uint8_t*)&dataHeader, sizeof(DataHeader));
    }

    return true;
}

bool LockFreeQueue::dequeue(void* output, size_t& out_size)
{
    flag_ = *buffer_;
    uint8_t old = 1;
    while (!flag_.compare_exchange_weak(old, 0)) {
        old = 1;
    }
    std::memcpy_s(&head_, buffer_ + 1, sizeof(size_t));
    std::memcpy_s(&tail_, buffer_ + 5, sizeof(size_t));
    if (head_ == tail_ && *(buffer_ + 9 + head_) == 0) {
        old = 0;
        while (!flag_.compare_exchange_weak(old, 1)) {
            old = 0;
        }
        return false;
    }

    size_t cur = getNextData();
    if (cur == SIZE_MAX) {
        std::cout << "LockFreeQueue error data" << std::endl;
        return false;
    }

    DataHeader* dataHeader{};
    if (tail_ > head_) {
        dataHeader = std::reinterpret_cast<DataHeader*>(buffer_ + 9 + cur);
        std::memcpy_s(output, buffer_ + 9 + cur + sizeof(DataHeader), dataHeader->data_size);
    } else {
        if (total_memory_size_ - cur > sizeof(DataHeader)) {
            dataHeader = std::reinterpret_cast<DataHeader*>(buffer_ + 9 + cur);
            if (total_memory_size_ - cur - sizeof(DataHeader) > dataHeader->data_size) {
                std::memcpy_s(output, buffer_ + 9 + cur + sizeof(DataHeader), dataHeader->data_size);
            } else {
                size_t first_part = total_memory_size_ - cur - sizeof(DataHeader);
                size_t second_part = dataHeader->data_size - first_part;
                std::memcpy_s(output, buffer_ + 9 + cur + sizeof(DataHeader), first_part);
                std::memcpy_s((uint8_t*)output + first_part, buffer_ + 9, second_part);
            }
        } else {
            size_t first_part = total_memory_size_ - cur;
            size_t second_part = sizeof(DataHeader) - first_part;
            std::memcpy_s(dataHeader, buffer_ + 9 + cur, first_part);
            std::memcpy_s(std::reinterpret_cast<uint8_t*>(dataHeader) + first_part, buffer_ + 9, second_part);
            std::memcpy_s(output, buffer_ + 9 + second_part, dataHeader->data_size);
        }
    }

    out_size = dataHeader->data_size;
    cur += out_size;
    std::memcpy_s(buffer_ + 1, &cur, sizeof(size_t));
    flag_ = *buffer_;
    old = 0;
    while (!flag_.compare_exchange_weak(old, 1)) {
        old = 0;
    }

    return true;
}

size_t LockFreeQueue::getNextData()
{
    DataHeader* header = nullptr;
    size_t cur = head_;
    while (cur != tail_) {
        uint8_t* ptr = buffer_ + cur;
        size_t remain_space = total_memory_size_ - cur;
        if (remain_space < sizeof(DataHeader)) {
            size_t first_part = remain_space;
            size_t second_part = sizeof(DataHeader) - first_part;

            if (second_part > (tail_ < head_ ? tail_ : 0)) {
                break;
            }

            std::memcpy_s(header, ptr, first_part);
            std::memcpy_s(std::reinterpret_cast<uint8_t*>(header) + first_part, buffer_ + 9, second_part);
        } else {
            std::memcpy_s(header, ptr, sizeof(DataHeader));
        }

        if (header->magic_prefix == 0xABCD1234) {
            size_t next = (cur + header->data_size) % total_memory_size_;
            if (next == tail_ || (tail_ > head_ && next < tail_) || (tail_ < head_ && (next > head_ || next < tail_))) {
                return cur;
            }
        }

        cur = (cur + 1) % total_memory_size_;
    }

    return SIZE_MAX;
}

}