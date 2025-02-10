// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "protocol.h"
#include <limits>
#include "serializer.h"

namespace Leaks {

void Protocol::Extractor::Feed(const std::string &msg)
{
    std::lock_guard<std::mutex> guard(mutex_);
    bytes_ += msg;
}

template <typename T>
bool Protocol::Extractor::Read(T &val)
{
    std::string buffer;
    if (!Read(sizeof(T), buffer)) {
        return false;
    }
    return Deserialize<T>(buffer, val);
}

bool Protocol::Extractor::Read(uint64_t size, std::string &buffer)
{
    std::lock_guard<std::mutex> guard(mutex_);
    uint64_t maxValue = std::numeric_limits<uint64_t>::max();
    if (maxValue - offset_ <= size || offset_ + size > bytes_.size()) {
        return false;
    }
    buffer = bytes_.substr(offset_, size);
    offset_ += size;

    if (offset_ > MAX_STRING_LEN) {
        DropUsedBytes();
    }
    return true;
}
uint64_t Protocol::Extractor::Size(void) const
{
    return bytes_.size() - offset_;
}

void Protocol::Extractor::DropUsedBytes(void)
{
    offset_ = std::min(offset_, bytes_.size());
    bytes_ = bytes_.substr(offset_);
    offset_ = 0UL;
}

Protocol::Protocol()
{
    extractor_ = std::make_shared<Extractor>();
}

void Protocol::Feed(std::string const &msg)
{
    if (msg.size() == 0) {
        return;
    }
    extractor_->Feed(msg);
    return;
}

Packet Protocol::GetPacket(void)
{
    thread_local static PacketHead head{PacketType::INVALID};
    if (head.type == PacketType::INVALID) {
        if (!extractor_->Read(head)) {
            return Packet {};
        }
    }

    Packet packet = GetPayLoad(head);
    if (packet.GetPacketHead().type != PacketType::INVALID) {
        head.type = PacketType::INVALID;
    }

    return packet;
}

Packet Protocol::GetPayLoad(PacketHead head)
{
    switch (head.type) {
        case PacketType::RECORD:
            return GetRecord();
        case PacketType::LOG:
            return GetLog();
        case PacketType::INVALID:
        default:
            return Packet{};
    }
}

bool Protocol::GetStringData(std::string &data)
{
    thread_local static uint64_t size{};
    if (size == 0UL) {
        if (!extractor_->Read(size)) {
            return false;
        }
    }
    if (extractor_->Size() < size) {
        return false;
    }
    if (!extractor_->Read(size, data)) {
        return false;
    }
    size = 0UL;
    return true;
}

Packet Protocol::GetRecord(void)
{
    EventRecord record{};
    if (!extractor_->Read(record)) {
        return Packet{};
    }
    auto packet = Packet(record);
    return packet;
}

Packet Protocol::GetLog(void)
{
    std::string buffer;
    return GetStringData(buffer) ? Packet(buffer) : Packet();
}

}
