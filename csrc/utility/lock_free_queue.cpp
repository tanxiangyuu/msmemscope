// Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

#include "lock_free_queue.h"
#include <cstring>
#include <iostream>
#include <cstddef>
#include <unistd.h>
#include "securec.h"

namespace Utility {

#define LEAKS_MAGIC_PREFIX 0xABCD1234
#define LEAKS_MSG_SEND_TIMEOUT_MS 3000
#define LEAKS_MSG_RECV_TIMEOUT_MS 50

struct DataHeader {
    /* 此处complate_flag需要放在最开头 */
    uint8_t complateFlag;
    uint8_t pad_[3];
    uint32_t magicPrefix;
    size_t dataSize;
    size_t clientId;
};

bool LockFreeQueue::ServerInit(uint64_t size)
{
    if (size <= sizeof(*this)) {
        std::cout << "[ERROR] Space for lock-free queue is too small." << std::endl;
        return false;
    }
    magic_ = LEAKS_MAGIC_PREFIX;
    size_ = size - sizeof(*this);
    new (&head_) std::atomic<uint64_t>(sizeof(*this));
    new (&tail_) std::atomic<uint64_t>(sizeof(*this));

    size_t msgBufLen = size - sizeof(*this);
    uint8_t *dataBuf = static_cast<uint8_t*>(static_cast<void*>(this)) + sizeof(*this);
    if (memset_s(dataBuf, msgBufLen, 0, msgBufLen)) {
        std::cout << "[ERROR] Failed to init lock-free queue buffer." << std::endl;
        return false;
    }

    return true;
}

bool LockFreeQueue::ClientInit()
{
    if (magic_ != LEAKS_MAGIC_PREFIX) {
        std::cout << "[ERROR] Failed to check self-ptr of lock-free queue." << std::endl;
        return false;
    }

    if (size_ <= sizeof(DataHeader) || head_.load(std::memory_order_relaxed) < sizeof(*this) ||
        tail_.load(std::memory_order_relaxed) < sizeof(*this)) {
        std::cout << "[ERROR] Bad lock-free queue head data." << std::endl;
        return false;
    }

    return true;
}

inline bool RingBufferMemcpyIn(void* buffer, size_t bufLen, uint64_t offset, const void* src, size_t length)
{
    if (length > bufLen) {
        return false;
    }
    uint64_t realOffset = offset % bufLen;
    if (realOffset + length <= bufLen) {
        return memcpy_s(static_cast<uint8_t*>(buffer) + realOffset, bufLen - realOffset, src, length) == 0;
    }
    size_t seg1 = bufLen - realOffset;
    if (memcpy_s(static_cast<uint8_t*>(buffer) + realOffset, seg1, src, seg1) != 0) {
        return false;
    }
    if (memcpy_s(buffer, realOffset, static_cast<const uint8_t*>(src) + seg1, length - seg1) != 0) {
        return false;
    }
    return true;
}

inline bool RingBufferMemcpyOut(void* dst, size_t length, const void* buffer, size_t bufLen, uint64_t offset)
{
    if (length > bufLen) {
        return false;
    }
    uint64_t realOffset = offset % bufLen;
    if (realOffset + length <= bufLen) {
        return memcpy_s(dst, length, static_cast<const uint8_t*>(buffer) + realOffset, length) == 0;
    }
    size_t seg1 = bufLen - realOffset;
    if (memcpy_s(dst, length, static_cast<const uint8_t*>(buffer) + realOffset, seg1) != 0) {
        return false;
    }
    if (memcpy_s(static_cast<uint8_t*>(dst) + seg1, length - seg1, buffer, length - seg1) != 0) {
        return false;
    }
    return true;
}

inline void RingBufferMemClear(void* buffer, size_t bufLen, uint64_t offset, size_t length)
{
    uint64_t realOffset = offset % bufLen;
    if (realOffset + length <= bufLen) {
        if (memset_s(static_cast<uint8_t*>(buffer) + realOffset, bufLen - realOffset, 0, length)) {
            std::cout << "[ERROR] failed to clear buffer offset:" << realOffset << std::endl;
        }
        return;
    }
    size_t seg1 = bufLen - realOffset;
    if (memset_s(static_cast<uint8_t*>(buffer) + realOffset, seg1, 0, seg1)) {
        std::cout << "[ERROR] failed to clear buffer offset:" << realOffset << std::endl;
    }
    if (memset_s(buffer, realOffset, 0, length - seg1)) {
        std::cout << "[ERROR] failed to clear buffer offset:0" << std::endl;
    }
    return;
}


bool LockFreeQueue::EnQueue(const void* data, size_t dataSize, size_t id)
{
    size_t totalSize = dataSize + sizeof(DataHeader);
    if (totalSize > size_) {
        return false;
    }
    uint64_t oriTail = tail_.fetch_add(totalSize, std::memory_order_relaxed);
    uint32_t cnt = LEAKS_MSG_SEND_TIMEOUT_MS * 1000;  // ms-->us
    while (oriTail + totalSize - head_.load(std::memory_order_relaxed) > size_) {
        if (--cnt == 0) {
            return false;
        }
        struct timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = 1000;  // 1us
        nanosleep(&ts, nullptr);
    }

    uint8_t *dataBuf = static_cast<uint8_t*>(static_cast<void*>(this)) + sizeof(*this);
    if (!RingBufferMemcpyIn(dataBuf, size_, oriTail + sizeof(DataHeader), data, dataSize)) {
        return false;
    }

    DataHeader head;
    head.complateFlag = 0;
    head.magicPrefix = LEAKS_MAGIC_PREFIX;
    head.dataSize = dataSize;
    head.clientId = id;
    if (!RingBufferMemcpyIn(dataBuf, size_, oriTail, &head, sizeof(DataHeader))) {
        return false;
    }

    /* 所有数据拷贝好后最后置falg */
    std::atomic_thread_fence(std::memory_order_release);
    static_cast<DataHeader*>(static_cast<void*>(dataBuf + oriTail % size_))->complateFlag = 1;
    return true;
}

bool LockFreeQueue::DeQueue(std::string& msg, size_t& id)
{
    if (IsEmpty()) {
        return false;
    }

    uint64_t offset = head_.load(std::memory_order_relaxed);
    uint8_t *dataBuf = static_cast<uint8_t*>(static_cast<void*>(this)) + sizeof(*this);
    DataHeader* dataHead = static_cast<DataHeader*>(static_cast<void*>(dataBuf + offset % size_));
    uint32_t cnt = LEAKS_MSG_RECV_TIMEOUT_MS * 1000;  // ms-->us
    while (dataHead->complateFlag != 1) {
        if (--cnt == 0) {
            return false;
        }
        struct timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = 1000;  // 1us
        nanosleep(&ts, nullptr);
    }
    std::atomic_thread_fence(std::memory_order_acquire);

    DataHeader head;
    if (!RingBufferMemcpyOut(&head, sizeof(head), dataBuf, size_, offset)) {
        return false;
    }
    if (head.magicPrefix != LEAKS_MAGIC_PREFIX) {
        return false;
    }
    msg.resize(head.dataSize);
    uint8_t *output = static_cast<uint8_t*>(static_cast<void*>((const_cast<char*>(msg.c_str()))));
    output[head.dataSize] = 0;
    if (!RingBufferMemcpyOut(output, head.dataSize, dataBuf, size_, offset + sizeof(head))) {
        return false;
    }
    RingBufferMemClear(dataBuf, size_, offset, sizeof(head) + head.dataSize);
    id = head.clientId;
    head_.fetch_add(sizeof(head) + head.dataSize, std::memory_order_relaxed);

    return true;
}

}