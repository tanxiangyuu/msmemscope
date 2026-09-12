/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#include "control_channel.h"

#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "analysis/event.h"
#include "analysis/event_dispatcher.h"
#include "control_channel/control_command_handler.h"
#include "control_channel/control_protocol.h"
#include "framework/process.h"
#include "nlohmann/json.hpp"
#include "utility/log.h"
#include "utility/utils.h"

namespace MemScope
{

namespace
{
// 全局POD(signal handler可安全访问;ctor显式赋值,规避init_array坑):
// SIGUSR1 handler只写g_wakePipeWriteFd唤醒监听线程;g_chActive会话中直接返回
volatile sig_atomic_t g_chActive = 0;
volatile sig_atomic_t g_wakePipeReadFd = -1;
volatile sig_atomic_t g_wakePipeWriteFd = -1;

// RESPONSE组帧与分片:output超限时按64KB边界切片多帧(除末帧外置CONT位),
// 每帧payload={"ok":bool,"output":<chunk>,"seq":<seq>};控制端逐帧拼接为完整响应。
// 序列化转义可能使帧超限(控制字符可放大6倍):减半chunk重试,末帧退化为非末帧
// kFrameChunkReserve=单chunk序列化转义余量(转义放大按字符计,预留256B足够;
// 极端仍超限时由减半重试兜底)
constexpr size_t kFrameChunkReserve = 256;
constexpr int kConnectTimeoutMs = 10000;  // 非阻塞connect的poll超时(对端异常时快速回IDLE)
bool SendResponse(int fd, uint16_t seq, bool ok, const std::string& output, std::string& error)
{
    const size_t chunkBudget = ControlProtocol::MAX_PAYLOAD_LEN - kFrameChunkReserve;
    size_t offset = 0;
    for (;;)
    {
        bool isLast = output.size() - offset <= chunkBudget;
        size_t chunkLen = isLast ? output.size() - offset : chunkBudget;
        std::string payload;
        for (;;)
        {
            const std::string chunk = output.substr(offset, chunkLen);
            nlohmann::json frame;
            frame["ok"] = ok;
            frame["output"] = chunk;
            frame["seq"] = seq;
            payload = frame.dump();
            if (payload.size() <= ControlProtocol::MAX_PAYLOAD_LEN)
            {
                break;
            }
            // 转义放大超限:末帧退化为非末帧,chunk减半重试
            isLast = false;
            chunkLen /= 2;
            if (chunkLen == 0)
            {
                error = "response chunk too large to serialize";
                return false;
            }
        }
        const uint16_t flags = isLast ? 0 : ControlProtocol::FLAG_CONT;
        if (!ControlProtocol::SendFrame(fd, ControlProtocol::MSG_RESPONSE, flags, seq, payload, error))
        {
            return false;
        }
        offset += chunkLen;
        if (isLast)
        {
            return true;
        }
    }
}
}  // namespace

ControlChannel& ControlChannel::GetInstance()
{
    static ControlChannel channel;
    return channel;
}

ControlChannel::ControlChannel()
{
    // 禁用env:关闭通道初始化(调试用,不写入用户资料)
    if (std::getenv(ControlProtocol::ENV_DISABLE_CONTROL_CHANNEL) != nullptr)
    {
        return;
    }
    int fds[2] = {-1, -1};
    if (::pipe2(fds, O_CLOEXEC) != 0)
    {
        return;
    }
    g_wakePipeReadFd = fds[0];
    g_wakePipeWriteFd = fds[1];
    g_chActive = 0;

    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_handler = &ControlChannel::OnSigUsr1;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    if (::sigaction(SIGUSR1, &sa, nullptr) != 0)
    {
        LOG_ERROR("[control channel] sigaction(SIGUSR1) failed: %s", std::strerror(errno));
        ::close(fds[0]);
        ::close(fds[1]);
        g_wakePipeReadFd = -1;
        g_wakePipeWriteFd = -1;
        return;
    }
    // 在pthread_create之前置位:create失败时父进程无监听线程,但fork child
    // 经OnAfterForkChild检查enabled_后自行重建pipe+线程(自救,不继承半初始化态)
    enabled_.store(true);

    // atfork子进程处理:关旧fd重建pipe并重启监听线程(子进程成为独立合法attach目标)
    ::pthread_atfork(nullptr, nullptr, &ControlChannel::OnAfterForkChild);

    pthread_t tid = 0;
    if (::pthread_create(&tid, nullptr, &ControlChannel::ListenThread, this) != 0)
    {
        LOG_ERROR("[control channel] pthread_create failed: %s", std::strerror(errno));
        ::close(fds[0]);
        ::close(fds[1]);
        g_wakePipeReadFd = -1;
        g_wakePipeWriteFd = -1;
        return;
    }
    ::pthread_setname_np(tid, "msmemscope_ctrl");  // 线程名=控制端规则4探针(恰好15字符)
}

// 库加载即激活:任何装配路径(钩子链/python API)加载libascend_leaks.so即启动控制通道
// (仿HostMemReportBoot先例;ctor只依赖libc安全的pipe2/sigaction/pthread_create,
// 均不触发hook链;禁用env在ctor内检查)
__attribute__((constructor)) static void ControlChannelBoot() { ControlChannel::GetInstance(); }

void ControlChannel::OnSigUsr1(int signo)
{
    (void)signo;
    // async-signal-safe:仅两条语句(会话中直接返回不重复唤醒;监听线程已阻塞在socket读)
    if (g_chActive != 0)
    {
        return;
    }
    const int wfd = g_wakePipeWriteFd;
    if (wfd >= 0)
    {
        const char byte = 'w';
        (void)::write(wfd, &byte, 1);
    }
}

void* ControlChannel::ListenThread(void* arg)
{
    ControlChannel* channel = static_cast<ControlChannel*>(arg);
    channel->Run();
    return nullptr;
}

void ControlChannel::OnAfterForkChild()
{
    // 子进程:旧pipe/socket fd属父进程,全部关闭后重建pipe;g_chActive清零;
    // 重启监听线程(本函数在fork的child侧执行,单线程上下文,直接调用)
    const int rfd = g_wakePipeReadFd;
    const int wfd = g_wakePipeWriteFd;
    if (rfd >= 0)
    {
        ::close(rfd);
    }
    if (wfd >= 0 && wfd != rfd)
    {
        ::close(wfd);
    }
    g_wakePipeReadFd = -1;
    g_wakePipeWriteFd = -1;
    g_chActive = 0;
    ControlChannel& channel = ControlChannel::GetInstance();
    if (!channel.enabled_.load())
    {
        return;  // 禁用env或ctor未完成:父进程本就没起监听线程
    }
    int fds[2] = {-1, -1};
    if (::pipe2(fds, O_CLOEXEC) != 0)
    {
        LOG_ERROR("[control channel] fork child pipe2 failed: %s", std::strerror(errno));
        return;
    }
    g_wakePipeReadFd = fds[0];
    g_wakePipeWriteFd = fds[1];
    pthread_t tid = 0;
    if (::pthread_create(&tid, nullptr, &ControlChannel::ListenThread, &channel) != 0)
    {
        LOG_ERROR("[control channel] fork child pthread_create failed: %s", std::strerror(errno));
        return;
    }
    ::pthread_setname_np(tid, "msmemscope_ctrl");
}

void ControlChannel::Run()
{
    // 注册CONTROL事件订阅(派发内仅登记,见ControlCommandHandler::HandleDispatch);
    // 订阅前不做任何派发,确保handler先于首帧到达就位。
    // fork child经OnAfterForkChild重启的监听线程:COW继承父进程订阅项,
    // subscribed_已置位则跳过Subscribe——避免继承到被持dispatcher锁时
    // (fork瞬间其他线程在派发)再取锁死锁
    if (!subscribed_.load())
    {
        EventDispatcher::GetInstance().Subscribe(
            SubscriberId::CONTROL_CHANNEL, {EventBaseType::CONTROL}, EventDispatcher::Priority::High,
            [](std::shared_ptr<EventBase>& event, MemoryState* state)
            { ControlCommandHandler::GetInstance().HandleDispatch(event, state); }, "control_channel");
        subscribed_.store(true);
    }

    for (;;)
    {
        // ---- IDLE:阻塞读唤醒pipe ----
        {
            char byte = 0;
            ssize_t n = ::read(g_wakePipeReadFd, &byte, 1);
            if (n < 0 && errno != EINTR)
            {
                // pipe读端损坏(不应发生):退避后继续等待
                ::usleep(100 * 1000);
                continue;
            }
        }

        // ---- CONNECTING:检查socket文件并connect ----
        const std::string sockPath = ControlProtocol::SocketPath(Utility::GetPid());
        if (::access(sockPath.c_str(), F_OK) != 0)
        {
            // 无socket文件(控制进程未bind或已删除):残留字节已消耗,回IDLE
            continue;
        }
        int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (fd < 0)
        {
            continue;
        }
        // 非阻塞connect+poll超时,避免对端不可达时挂死监听线程
        struct sockaddr_un addr;
        std::memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        std::strncpy(addr.sun_path, sockPath.c_str(), sizeof(addr.sun_path) - 1);
        const int flags = ::fcntl(fd, F_GETFL, 0);
        ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        bool connected = false;
        if (::connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) == 0)
        {
            connected = true;
        }
        else if (errno == EINPROGRESS)
        {
            struct pollfd pfd = {fd, POLLOUT, 0};
            const int pr = ::poll(&pfd, 1, kConnectTimeoutMs);
            int soError = 0;
            socklen_t soLen = sizeof(soError);
            if (pr > 0 && ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &soError, &soLen) == 0 && soError == 0)
            {
                connected = true;
            }
        }
        ::fcntl(fd, F_SETFL, flags);  // 恢复阻塞
        if (!connected)
        {
            ::close(fd);
            continue;  // 对端已退出(ECONNREFUSED等):回IDLE等待下次唤醒
        }

        // ---- 握手:发送REGISTER ----
        nlohmann::json reg;
        reg["pid"] = Utility::GetPid();
        reg["proto"] = ControlProtocol::PROTO_VERSION;
        std::string sendErr;
        if (!ControlProtocol::SendFrame(fd, ControlProtocol::MSG_REGISTER, 0, 0, reg.dump(), sendErr))
        {
            LOG_ERROR("[control channel] REGISTER failed: %s", sendErr.c_str());
            ::close(fd);
            continue;
        }

        // ---- ACTIVE:会话循环(EOF/错误回IDLE) ----
        g_chActive = 1;
        for (;;)
        {
            struct pollfd pfd = {fd, POLLIN, 0};
            const int pr = ::poll(&pfd, 1, -1);
            if (pr < 0)
            {
                if (errno == EINTR)
                {
                    continue;
                }
                break;  // poll异常(不应发生):回IDLE
            }
            if (pr == 0)
            {
                continue;
            }
            if ((pfd.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0)
            {
                break;  // 控制进程退出/会话异常:回IDLE
            }
            std::vector<uint8_t> payload;
            uint16_t seq = 0;
            uint8_t msgType = 0;
            uint16_t flags = 0;
            std::string err;
            if (!ControlProtocol::ReadFrame(fd, payload, seq, msgType, flags, err))
            {
                LOG_INFO("[control channel] session ended: %s", err.c_str());
                break;
            }
            if (msgType != ControlProtocol::MSG_CONTROL)
            {
                LOG_ERROR("[control channel] unexpected msg_type=%u, session dropped", msgType);
                break;  // 非法消息:断开会话(不静默吞)
            }
            nlohmann::json parsed;
            try
            {
                parsed = nlohmann::json::parse(payload.begin(), payload.end());
            }
            catch (const std::exception&)
            {
                LOG_ERROR("[control channel] invalid JSON payload, session dropped");
                break;
            }
            const std::string cmd = parsed.value("cmd", "");
            const std::string param = parsed.value("param", "");
            if (cmd.empty())
            {
                LOG_ERROR("[control channel] empty cmd, session dropped");
                break;
            }

            // 构造ControlEvent并派发:派发内仅登记(白名单校验+复制cmd/param),
            // 命令执行统一在SendEvent返回后(派发锁已释放,规避dispatcher非递归锁重入)
            std::shared_ptr<ControlEvent> ev = std::make_shared<ControlEvent>();
            ev->cmd = cmd;
            ev->param = param;
            Process::GetInstance().SendEvent(ev);
            ControlCommandHandler::GetInstance().Execute(ev);

            // RESPONSE回带请求seq与业务侧ok位;output超限按64KB切片多帧(CONT位)
            std::string respErr;
            if (!SendResponse(fd, seq, ev->ok, ev->output, respErr))
            {
                LOG_INFO("[control channel] response send failed: %s", respErr.c_str());
                break;
            }
        }
        g_chActive = 0;
        ::close(fd);
        // 回IDLE:等待下次SIGUSR1唤醒
    }
}

}  // namespace MemScope
