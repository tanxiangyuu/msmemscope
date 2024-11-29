// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef FRAMEWORK_PROTOCOL_H
#define FRAMEWORK_PROTOCOL_H

#include <string>
#include <memory>
#include <mutex>
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

class Protocol::Extractor {
public:
    Extractor() = default;
    ~Extractor() = default;
    inline void Feed(const std::string &msg);
    template<typename T>
    inline bool Read(T &val);
    inline bool Read(uint64_t size, std::string &buffer);

private:
    static constexpr uint64_t MAX_STRING_LEN = 1024UL;
    inline void DropUsedBytes(void);

private:
    std::string bytes_;
    std::string::size_type offset_{0UL};
    std::mutex mutex_;
};

}

#endif