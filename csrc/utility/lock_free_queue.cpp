// Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

#include "lock_free_queue.h"
#include <cstring>
#include <iostream>
#include <cstddef>
#include "securec.h"


namespace Utility {


void print_hex(const uint8_t *ptr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        printf("%02x ", ptr[i]);  // %02x 表示 2 位十六进制，不足补零
    }
    printf("\n");
}

LockFreeQueue::LockFreeQueue(size_t total_memory_size, uint8_t* buffer):total_memory_size_(total_memory_size),
    buffer_(buffer), head_(0), tail_(0), free_space_(0), flag_(nullptr), dataHeader_(new DataHeader(0, 0)),
    outputBuf_(new uint8_t[OUTPUT_BUF_SIZE]) { }

bool LockFreeQueue::enqueue(const void* data, size_t data_size, size_t id)
{
    // static count = 0;
    // ++count;
    flag_ = reinterpret_cast<std::atomic<uint8_t>*>(buffer_);
    uint8_t old = 0;
    while (!flag_->compare_exchange_weak(old, 1)) {
        old = 0;
    }
    if (memcpy_s(&head_, sizeof(size_t), buffer_ + 1, sizeof(size_t))) {
        return false;
    }
    if (memcpy_s(&tail_, sizeof(size_t), buffer_ + 1 + sizeof(size_t), sizeof(size_t))) {
        return false;
    }
    
    free_space_ = tail_ >= head_ ? total_memory_size_ - (tail_ - head_) : head_ - tail_;

    size_t required_size = data_size + sizeof(DataHeader);
    
    if (free_space_ < required_size) {
        old = 1;
        while (!flag_->compare_exchange_weak(old, 0)) {
            old = 1;
        }
        return false;
    }

    bool needs_truncation = tail_ + required_size > total_memory_size_;

    size_t old_tail = tail_;

    tail_ = needs_truncation ? required_size - (total_memory_size_ - tail_) : tail_ + required_size;
    if (memcpy_s(buffer_ + 1 + sizeof(size_t), total_memory_size_ + sizeof(size_t), &tail_, sizeof(size_t))) {
        return false;
    }
    old = 1;
    while (!flag_->compare_exchange_weak(old, 0)) {
        old = 1;
    }

    DataHeader dataHeader(data_size, id);

    if (needs_truncation) {
        size_t tail_size = total_memory_size_ - old_tail;
        if (tail_size < sizeof(DataHeader)) {
            if (memcpy_s(buffer_ + 1 + 2 * sizeof(size_t) + (sizeof(DataHeader) - tail_size), total_memory_size_ - (sizeof(DataHeader) - tail_size), data, data_size)) {
                return false;
            }
            if (memcpy_s(buffer_ + 1 + 2 * sizeof(size_t) + old_tail, total_memory_size_ - old_tail, reinterpret_cast<uint8_t*>(&dataHeader), tail_size)) {
                return false;
            }
            if (memcpy_s(buffer_ + 1 + 2 * sizeof(size_t), total_memory_size_, (reinterpret_cast<uint8_t*>(&dataHeader)) + tail_size, sizeof(DataHeader) - tail_size)) {
                return false;
            }
            // std::cout << "1:" << dataHeader.magic_prefix << std::endl;
            // print_hex(buffer_ + 1 + 2 * sizeof(size_t) + old_tail, tail_size);
            // print_hex(buffer_ + 1 + 2 * sizeof(size_t), sizeof(DataHeader) - tail_size);
        } else {
            if (memcpy_s(buffer_ + 1 + 2 * sizeof(size_t) + old_tail + sizeof(DataHeader), total_memory_size_ - old_tail - sizeof(DataHeader), data, tail_size - sizeof(DataHeader))) {
                return false;
            }
            if (memcpy_s(buffer_ + 1 + 2 * sizeof(size_t), total_memory_size_, reinterpret_cast<const uint8_t*>(data) + (tail_size - sizeof(DataHeader)), data_size - (tail_size - sizeof(DataHeader)))) {
                return false;
            }
            if (memcpy_s(buffer_ + 1 + 2 * sizeof(size_t) + old_tail, total_memory_size_ - old_tail, &dataHeader, sizeof(DataHeader))) {
                return false;
            }
            // std::cout << "2:" <<dataHeader.magic_prefix << std::endl;
            // print_hex(buffer_ + 1 + 2 * sizeof(size_t) + old_tail, sizeof(DataHeader));
        }
    } else {
        if (memcpy_s(buffer_ + 1 + 2 * sizeof(size_t) + old_tail + sizeof(DataHeader), total_memory_size_ - old_tail - sizeof(DataHeader), data, data_size)) {
            return false;
        }
        if (memcpy_s(buffer_ + 1 + 2 * sizeof(size_t) + old_tail, total_memory_size_ - old_tail, &dataHeader, sizeof(DataHeader))) {
            return false;
        }
        // std::cout << "3:" <<dataHeader.magic_prefix << std::endl;
        // print_hex(buffer_ + 1 + 2 * sizeof(size_t) + old_tail, sizeof(DataHeader));
    }

    return true;
}

bool LockFreeQueue::dequeue(void** output, size_t& out_size, size_t& id)
{
    if (memcpy_s(&head_, sizeof(size_t), buffer_ + 1, sizeof(size_t))) {
        return false;
    }
    if (memcpy_s(&tail_, sizeof(size_t), buffer_ + 1 + sizeof(size_t), sizeof(size_t))) {
        return false;
    }
    if (head_ == tail_ && *(buffer_ + 1 + 2 * sizeof(size_t) + head_) == 0) {
        return false;
    }

    size_t cur = getNextData();
    if (cur == SIZE_MAX) {
        
        return false;
    }

    out_size = dataHeader_->data_size;
    id = dataHeader_->clientId;
    if (tail_ > head_) {
        if (memcpy_s(outputBuf_, OUTPUT_BUF_SIZE, buffer_ + 1 + 2 * sizeof(size_t) + cur + sizeof(DataHeader), out_size)) {
            return false;
        }
        if (memset_s(buffer_ + 1 + 2 * sizeof(size_t) + cur, total_memory_size_, 0, out_size + sizeof(DataHeader))) {
            return false;
        }
    } else {
        if (total_memory_size_ - cur > sizeof(DataHeader)) {
            if (total_memory_size_ - cur - sizeof(DataHeader) > out_size) {
                if (memcpy_s(outputBuf_, OUTPUT_BUF_SIZE, buffer_ + 1 + 2 * sizeof(size_t) + cur + sizeof(DataHeader), out_size)) {
                    return false;
                }
                if (memset_s(buffer_ + 1 + 2 * sizeof(size_t) + cur, total_memory_size_ - cur, 0, out_size + sizeof(DataHeader))) {
                    return false;
                }
            } else {
                size_t first_part = total_memory_size_ - cur - sizeof(DataHeader);
                size_t second_part = out_size - first_part;
                if (memcpy_s(outputBuf_, OUTPUT_BUF_SIZE, buffer_ + 1 + 2 * sizeof(size_t) + cur + sizeof(DataHeader), first_part)) {
                    return false;
                }
                if (memset_s(buffer_ + 1 + 2 * sizeof(size_t) + cur + sizeof(DataHeader), total_memory_size_ - cur - sizeof(DataHeader), 0, first_part)) {
                    return false;
                }
                if (memcpy_s(outputBuf_ + first_part, OUTPUT_BUF_SIZE, buffer_ + 1 + 2 * sizeof(size_t), second_part)) {
                    return false;
                }
                if (memset_s(buffer_ + 1 + 2 * sizeof(size_t), total_memory_size_, 0, second_part)) {
                    return false;
                }
            }
        } else {
            size_t first_part = total_memory_size_ - cur;
            size_t second_part = sizeof(DataHeader) - first_part;
            if (memcpy_s(outputBuf_, OUTPUT_BUF_SIZE, buffer_ + 1 + 2 * sizeof(size_t) + second_part, out_size)) {
                return false;
            }
            if (memset_s(buffer_ + 1 + 2 * sizeof(size_t) + second_part, total_memory_size_ - second_part, 0, out_size)) {
                return false;
            }
        }
    }

    *output = outputBuf_;

    cur = (cur + out_size + sizeof(DataHeader)) % (total_memory_size_);
    flag_ = reinterpret_cast<std::atomic<uint8_t>*>(buffer_);
    uint8_t old = 0;
    while (!flag_->compare_exchange_weak(old, 1)) {
        old = 0;
    }
    if (memcpy_s(buffer_ + 1, total_memory_size_ + 2 * sizeof(size_t), &cur, sizeof(size_t))) {
        return false;
    }

    old = 1;
    while (!flag_->compare_exchange_weak(old, 0)) {
        old = 1;
    }

    return true;
}

size_t LockFreeQueue::getNextData()
{
    size_t cur = head_;
    uint8_t* ptr = buffer_ + 1 + 2 * sizeof(size_t) + cur;
    size_t remain_space = total_memory_size_ - cur;
    if (remain_space < sizeof(DataHeader)) {
        size_t first_part = remain_space;
        size_t second_part = sizeof(DataHeader) - first_part;

        if (second_part > (tail_ < head_ ? tail_ : 0)) {
            return SIZE_MAX;
        }

        if (memcpy_s(dataHeader_, sizeof(DataHeader), ptr, first_part)) {
            return SIZE_MAX;
        }
        if (memcpy_s(reinterpret_cast<uint8_t*>(dataHeader_) + first_part, second_part, buffer_ + 9, second_part)) {
            return SIZE_MAX;
        }
    } else {
        if (memcpy_s(dataHeader_, sizeof(DataHeader), ptr, sizeof(DataHeader))) {
            return SIZE_MAX;
        }
    }
    if (dataHeader_->magic_prefix == 0xABCD1234) {
        size_t next = (cur + sizeof(DataHeader) + dataHeader_->data_size) % total_memory_size_;
        if (next == tail_ || (tail_ > head_ && next < tail_) || (tail_ < head_ && (next > head_ || next < tail_))) {
            return cur;
        }
    } else {
        std::cout << "head data is broken\n";
    }

    return SIZE_MAX;
}

}