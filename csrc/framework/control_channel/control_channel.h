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

#ifndef CONTROL_CHANNEL_H
#define CONTROL_CHANNEL_H

#include <atomic>
#include <cstdint>

namespace MemScope
{

/*
 * 进程外控制通道(业务进程侧): 随libascend_leaks.so加载即激活(任何装配路径),常驻
 * 1条监听线程+1个pipe fd,零CPU。SIGUSR1 handler仅做async-signal-safe的pipe写入;
 * ACTIVE会话中handler直接返回。收帧→构造ControlEvent→Process::SendEvent(派发内仅
 * 登记)→派发返回后Execute(派发外执行,规避dispatcher非递归锁重入)→RESPONSE回传。
 */
class ControlChannel
{
   public:
    static ControlChannel& GetInstance();

    // 通道是否已激活(禁用env未设且ctor已完成):控制端规则4线程名探针的语义前提
    bool IsEnabled() const { return enabled_.load(); }

   private:
    ControlChannel();
    ~ControlChannel() = default;
    ControlChannel(const ControlChannel&) = delete;
    ControlChannel& operator=(const ControlChannel&) = delete;

    // SIGUSR1处理:仅写pipe唤醒监听线程(全局POD,全async-signal-safe)
    static void OnSigUsr1(int signo);
    static void* ListenThread(void* arg);
    void Run();  // 监听线程状态机:IDLE(阻塞读pipe)→CONNECTING→ACTIVE(会话循环)→回IDLE

    // fork子进程处理:关旧pipe/会话fd重建pipe并重启监听线程(仿host_mem_hooks ForkPrepare
    // 先例:子进程成为独立合法attach目标)
    static void OnAfterForkChild();

   private:
    std::atomic<bool> enabled_{false};     // sigaction成功即置位(create失败时fork child可自救重建)
    std::atomic<bool> subscribed_{false};  // 订阅已登记:fork child COW继承订阅项跳过重复Subscribe
};

}  // namespace MemScope

#endif
