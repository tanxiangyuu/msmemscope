// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef __CORE_REMOTE_PROCESS_H__
#define __CORE_REMOTE_PROCESS_H__

#include <string>
#include "host_injection/core/Communication.h"

// ServerProcess类是远端进程的抽象，主要是工具所在的进程
// 该类主要提供与LocalProcess协同的能力
class ServerProcess {
public:
    explicit ServerProcess(CommType type);
    ~ServerProcess();

    /** 启动服务端
     * @description 启动服务端，在此之前可以设置 SetMsgHandlerHook
     * 回调，防止回调设置前一些客户端已经连接
     */
    void Start();

    /** 等待 LocalProcess 消息
     * @param clientIdx 要等待的 LocalProcess 客户端编号，此编号代表客户端的连接顺序
     * 从 0 开始计数
     * @param msg 接收到的消息
     * @param timeOut 等待超时时间，单位毫秒
     * @return >= 0 接收成功，并且返回值表示实际接收到的字节数
     *         <  0 接收失败
     */
    int Wait(std::size_t clientId, std::string& msg);

    /** 向 LocalProcess 发送消息
     * @param clientIdx 要发送的 LocalProcess 客户端编号，此编号代表客户端的连接顺序
     * 从 0 开始计数
     * @param msg 要发送的消息
     * @return >= 0 发送成功，并且返回值表示实际发送的字节数
     *         <  0 发送失败
     */
    int Notify(std::size_t clientId, const std::string& msg);

    /** 设置消息接收通知的回调函数
     * @param func 通知回调函数
     */
    void SetMsgHandlerHook(ClientMsgHandlerHook &&hook);

    /** 设置客户端连接后处理的回调函数
     * @param func 客户端连接回调函数
     */
    void SetClientConnectHook(ClientConnectHook &&hook);

private:
    Server* server_ = nullptr;
};

#endif // __CORE_REMOTE_PROCESS_H__
