/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

#include "control_protocol.h"

#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>

namespace MemScope
{
namespace ControlProtocol
{

std::string SocketPath(uint64_t pid)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "/tmp/msmemscope_socket_%llu", static_cast<unsigned long long>(pid));
    return std::string(buf);
}

namespace
{
bool Contains(const char* const* words, size_t count, const std::string& token)
{
    for (size_t i = 0; i < count; ++i)
    {
        if (token == words[i])
        {
            return true;
        }
    }
    return false;
}

constexpr int kSendTimeoutMs = 10000;  // 帧发送poll超时(会话断开兜底)
}  // namespace

bool ReadFrame(int fd, std::vector<uint8_t>& payloadOut, uint16_t& seqOut, uint8_t& msgTypeOut, uint16_t& flagsOut,
               std::string& error)
{
    uint8_t header[HEADER_LEN];
    size_t got = 0;
    while (got < sizeof(header))
    {
        ssize_t n = ::recv(fd, header + got, sizeof(header) - got, 0);
        if (n > 0)
        {
            got += static_cast<size_t>(n);
            continue;
        }
        if (n < 0 && (errno == EINTR || errno == EAGAIN))
        {
            continue;
        }
        error = n == 0 ? "peer closed" : std::string("recv header: ") + std::strerror(errno);
        return false;
    }
    const uint8_t version = header[0];
    msgTypeOut = header[1];
    flagsOut = static_cast<uint16_t>(header[2]) | (static_cast<uint16_t>(header[3]) << 8);
    seqOut = static_cast<uint16_t>(header[4]) | (static_cast<uint16_t>(header[5]) << 8);
    uint32_t payloadLen = 0;
    for (size_t i = 0; i < 4; ++i)
    {
        payloadLen |= static_cast<uint32_t>(header[6 + i]) << (8 * i);
    }
    if (version != PROTO_VERSION)
    {
        error = "unsupported protocol version";
        return false;
    }
    if (payloadLen > MAX_PAYLOAD_LEN)
    {
        error = "payload too large";
        return false;
    }
    payloadOut.resize(payloadLen);
    got = 0;
    while (got < payloadLen)
    {
        ssize_t n = ::recv(fd, payloadOut.data() + got, payloadLen - got, 0);
        if (n > 0)
        {
            got += static_cast<size_t>(n);
            continue;
        }
        if (n < 0 && (errno == EINTR || errno == EAGAIN))
        {
            continue;
        }
        error = n == 0 ? "peer closed" : std::string("recv payload: ") + std::strerror(errno);
        return false;
    }
    return true;
}

bool SendFrame(int fd, uint8_t msgType, uint16_t flags, uint16_t seq, const std::string& payload, std::string& error)
{
    if (payload.size() > MAX_PAYLOAD_LEN)
    {
        error = "frame payload exceeds limit";
        return false;
    }
    uint8_t header[HEADER_LEN];
    std::memset(header, 0, sizeof(header));
    header[0] = PROTO_VERSION;
    header[1] = msgType;
    header[2] = static_cast<uint8_t>(flags & 0xFF);
    header[3] = static_cast<uint8_t>((flags >> 8) & 0xFF);
    header[4] = static_cast<uint8_t>(seq & 0xFF);
    header[5] = static_cast<uint8_t>((seq >> 8) & 0xFF);
    const uint32_t len = static_cast<uint32_t>(payload.size());
    for (size_t i = 0; i < 4; ++i)
    {
        header[6 + i] = static_cast<uint8_t>((len >> (8 * i)) & 0xFF);
    }

    size_t sent = 0;
    const size_t total = sizeof(header) + payload.size();
    while (sent < total)
    {
        // 每轮发送前poll可写:对端半开(只FIN不读)时首轮通过后写仍会阻塞——
        // 每轮poll把单次send阻塞上限钉死在kSendTimeoutMs
        struct pollfd pfd = {fd, POLLOUT, 0};
        const int pr = ::poll(&pfd, 1, kSendTimeoutMs);
        if (pr <= 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
        {
            error = pr == 0 ? "send poll timeout" : "send poll failed";
            return false;
        }
        // 先发送header再发送payload
        const size_t chunk = (sent < sizeof(header)) ? (sizeof(header) - sent) : (total - sent);
        const void* buf = sent < sizeof(header) ? static_cast<const void*>(header + sent)
                                                : static_cast<const void*>(payload.data() + sent - sizeof(header));
        ssize_t n = ::send(fd, buf, chunk, MSG_NOSIGNAL);
        if (n > 0)
        {
            sent += static_cast<size_t>(n);
            continue;
        }
        if (n < 0 && (errno == EINTR || errno == EAGAIN))
        {
            continue;
        }
        error = n < 0 ? std::string("send: ") + std::strerror(errno) : "send: peer closed";
        return false;
    }
    return true;
}

bool IsValidControlWord(const std::vector<std::string>& tokens)
{
    if (tokens.empty())
    {
        return false;
    }
    const std::string& head = tokens[0];
    if (tokens.size() == 1)
    {
        return Contains(COMMON_WORDS, sizeof(COMMON_WORDS) / sizeof(COMMON_WORDS[0]), head);
    }
    if (head == "display" && tokens.size() == 2)
    {
        return Contains(DISPLAY_WORDS, sizeof(DISPLAY_WORDS) / sizeof(DISPLAY_WORDS[0]), tokens[1]);
    }
    if (head == "display" && tokens.size() >= 3)
    {
        // 主干2级词校验,其后参数由业务侧解析(display memory block --pool hal等带参命令)
        if (tokens[1] == "memory")
        {
            return Contains(MEMORY_WORDS, sizeof(MEMORY_WORDS) / sizeof(MEMORY_WORDS[0]), tokens[2]);
        }
        if (tokens[1] == "host_leak" && tokens[2] == "summary")
        {
            return true;
        }
        return false;
    }
    if (head == "set" && tokens.size() >= 2)
    {
        // set config <param...>: 二级词校验,其后参数由set config解析
        return Contains(SET_WORDS, sizeof(SET_WORDS) / sizeof(SET_WORDS[0]), tokens[1]);
    }
    return false;
}

}  // namespace ControlProtocol
}  // namespace MemScope
