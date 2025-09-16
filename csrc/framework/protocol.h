// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef FRAMEWORK_PROTOCOL_H
#define FRAMEWORK_PROTOCOL_H

#include <string>
#include <memory>
#include <mutex>
#include "config_info.h"
#include "record_info.h"
#include "serializer.h"

namespace Leaks {

constexpr size_t MAX_STRING_LEN = 65536;

enum class PacketType : uint8_t {
    RECORD = 0,
    LOG,
    INVALID
};

struct PacketHead {
    PacketType type;
    size_t length;
};

struct LogInfo {
    uint64_t len;
    char *buf;
};

union PacketBody {
    Record record;
    LogInfo log;
};

class Packet {
public:
    Packet(void) : head_{PacketType::INVALID}, body_{}
    {}

    Packet(Packet &&rhs) : head_{rhs.head_}, body_{rhs.body_}
    {
        if (rhs.head_.type == PacketType::LOG) {
            body_.log.buf = rhs.body_.log.buf;
            rhs.body_.log.buf = nullptr;
        } else if (rhs.head_.type == PacketType::RECORD) {
            body_.record.callStackInfo.cStack = rhs.body_.record.callStackInfo.cStack;
            rhs.body_.record.callStackInfo.cStack = nullptr;
            body_.record.callStackInfo.pyStack = rhs.body_.record.callStackInfo.pyStack;
            rhs.body_.record.callStackInfo.pyStack = nullptr;
        }
    }

    ~Packet(void)
    {
        if (head_.type == PacketType::RECORD) {
            delete[] body_.record.callStackInfo.cStack;
            delete[] body_.record.callStackInfo.pyStack;
        } else if (head_.type == PacketType::LOG) {
            delete[] body_.log.buf;
        }
    }
    explicit Packet(EventRecord const &record, std::string &pyStack, std::string &cStack)
    {
        head_.type = PacketType::RECORD;
        head_.length = sizeof(head_.type);
        uint64_t cLen = std::min(cStack.size(), MAX_STRING_LEN);
        uint64_t pyLen = std::min(pyStack.size(), MAX_STRING_LEN);
        body_.record.eventRecord = record;
        body_.record.callStackInfo.cStack = new char[cLen];
        body_.record.callStackInfo.cLen = cLen;
        body_.record.callStackInfo.pyStack = new char[pyLen];
        body_.record.callStackInfo.pyLen = pyLen;
        pyStack.copy(body_.record.callStackInfo.pyStack, pyLen);
        cStack.copy(body_.record.callStackInfo.cStack, cLen);
    }
    explicit Packet(std::string log)
    {
        uint64_t len = std::min(log.size(), MAX_STRING_LEN);
        head_.type = PacketType::LOG;
        head_.length = sizeof(head_.type);
        body_.log.len = len;
        body_.log.buf = new char[len];
        log.copy(body_.log.buf, len);
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
    class Extractor {
    public:
        Extractor() = default;
        ~Extractor() = default;
        void Feed(const std::string &msg);
        uint64_t Size(void) const;
        template<typename T>
        bool Read(T &val);
        bool Read(uint64_t size, std::string &buffer);
        template<typename T>
        bool View(T &val);
        bool View(uint64_t size, std::string &buffer);

    private:
        static constexpr uint64_t MAX_STRING_LEN = 1024UL * 1024UL;
        void DropUsedBytes(void);

    private:
        std::string bytes_;
        std::string::size_type offset_{0UL};
        std::mutex mutex_;
    };

public:
    Protocol();
    ~Protocol() = default;
    void Feed(std::string const &msg);
    inline uint64_t Size() const {return extractor_->Size();}
    template<typename T>
    inline bool View(T &val) {return extractor_->View(val);}
    template<typename T>
    inline bool Read(T &val) {return extractor_->Read(val);}
    bool GetStringData(std::string &data, uint64_t size);
    Packet GetPacket(void);
private:
    Packet GetPayLoad(PacketHead head);
    Packet GetRecord(void);
    Packet GetLog(uint64_t len);
    std::shared_ptr<Extractor> extractor_;
};

template <typename T>
bool Protocol::Extractor::Read(T &val)
{
    std::string buffer;
    if (!Read(sizeof(T), buffer)) {
        return false;
    }
    return Deserialize<T>(buffer, val);
}

template <typename T>
bool Protocol::Extractor::View(T &val)
{
    std::string buffer;
    if (!View(sizeof(T), buffer)) {
        return false;
    }
    return Deserialize<T>(buffer, val);
}

}

#endif