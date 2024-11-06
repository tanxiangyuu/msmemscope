// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef FRAMEWORK_PROTOCOL_H
#define FRAMEWORK_PROTOCOL_H

#include <string>
#include <memory>
#include "config_info.h"
#include "record_info.h"

namespace Leaks {

enum class PacketType : uint8_t {
    RECORD = 0,
    INVALID
};

struct PacketHead {
    PacketType type;
};

using PacketBody = EventRecord;

class Packet {
public:
    Packet(void) : head_{PacketType::INVALID}, body_{} { }
    explicit Packet(EventRecord const &record)
    {
        head_.type = PacketType::RECORD;
        body_ = record;
    }
    PacketHead GetPacketHead(void) const
    {
        return head_;
    }
    PacketBody const &GetPacketBody(void) const
    {
        return body_;
    }
private:
    PacketHead head_;
    PacketBody body_;
};

// Protocol类接收数据，根据协议解包，后期可根据分析算法的不同进行扩展
class Protocol {
public:
    Protocol();
    ~Protocol() = default;
    void Feed(std::string const &msg);
    Packet GetPacket(void);
private:
    Packet GetPayLoad(PacketHead head);
    Packet GetRecord(void);

    class Extractor;
    std::shared_ptr<Extractor> extractor_;
};

}

#endif