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

#ifndef CONTROL_PROTOCOL_H
#define CONTROL_PROTOCOL_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace MemScope
{
// 进程外控制通道协议定义(控制端AttachController与业务侧ControlChannel共用)。
// 帧格式(小端):
//   [0]     version u8(当前=1)
//   [1]     msg_type u8: 0x01 REGISTER / 0x02 CONTROL / 0x03 RESPONSE
//   [2..3]  flags u8×2: bit0=CONT(非末帧,仅RESPONSE分片时置位)
//   [4..5]  seq u16 LE: CONTROL由控制端单调递增,RESPONSE回带同号;REGISTER固定0
//   [6..9]  payload_len u32 LE
//   [10..]  payload(JSON,单帧≤64KB)
namespace ControlProtocol
{
constexpr uint8_t PROTO_VERSION = 1;
constexpr uint8_t MSG_REGISTER = 0x01;  // b→c {"pid":12345,"proto":1}
constexpr uint8_t MSG_CONTROL = 0x02;   // c→b {"cmd":"start","param":""}
constexpr uint8_t MSG_RESPONSE = 0x03;  // b→c {"ok":true,"output":"..."}
constexpr size_t HEADER_LEN = 10;
constexpr size_t MAX_PAYLOAD_LEN = 64 * 1024;  // 单帧payload上限64KB
constexpr uint16_t FLAG_CONT = 0x0001;         // 非末帧(仅RESPONSE分片)

// socket文件:/tmp(sticky)下固定命名,0600;生命周期=握手期(控制端收到REGISTER即删除)
std::string SocketPath(uint64_t pid);

// 调试env(不写入用户资料):关闭业务侧通道初始化
constexpr const char* ENV_DISABLE_CONTROL_CHANNEL = "MSMEMSCOPE_DISABLE_CONTROL_CHANNEL";
// 控制端accept等待超时(ms,默认30000);调试env可调(短超时暴露already attached场景)
constexpr const char* ENV_ATTACH_TIMEOUT_MS = "MSMEMSCOPE_ATTACH_TIMEOUT_MS";

// 帧编解码(控制端AttachController与业务侧ControlChannel共用,两端同源编译):
// ReadFrame收一帧(头+payload);SendFrame发一帧(poll超时+MSG_NOSIGNAL防SIGPIPE)。
// 失败返回false并带错误信息(跨进程无共享状态,纯函数式编解码)
bool ReadFrame(int fd, std::vector<uint8_t>& payloadOut, uint16_t& seqOut, uint8_t& msgTypeOut, uint16_t& flagsOut,
               std::string& error);
bool SendFrame(int fd, uint8_t msgType, uint16_t flags, uint16_t seq, const std::string& payload, std::string& error);

// 控制字白名单(控制端与业务端共用同一常量表,两端独立校验;help/exit仅控制端本地命令)
constexpr const char* const COMMON_WORDS[] = {"start", "stop", "step"};
constexpr const char* const DISPLAY_WORDS[] = {"hook", "analyzer", "config", "memory", "host_leak"};
constexpr const char* const MEMORY_WORDS[] = {"summary", "block"};
constexpr const char* const SET_WORDS[] = {"config"};

// 空白分词后的token序列是否为合法控制字:
// 单token(start/stop/step) / display <word> / display memory <word> /
// display host_leak summary / set config(其后参数不参与校验,由set config解析)
bool IsValidControlWord(const std::vector<std::string>& tokens);
}  // namespace ControlProtocol

}  // namespace MemScope

#endif
