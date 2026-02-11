/* ------------------------------------------------------------------------- * This file is part of the MindStudio project.
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
 * ------------------------------------------------------------------------- */

#ifndef OOM_HANDLER_H
#define OOM_HANDLER_H

#include <mutex>
#include <queue>
#include "record_info.h"

namespace MemScope {

/*
 * OOMHandler类主要功能：
 * 1. 管理OOM调用栈的存储和获取
 * 2. 提供线程安全的接口，确保OOM调用栈的正确传递
 */
class OOMHandler {
public:
    static OOMHandler& Instance();
    
    // 设置OOM调用栈
    void SetOOMStack(const CallStackString& stack);
    
    // 获取OOM调用栈
    CallStackString GetOOMStack();

private:
    OOMHandler() = default;
    ~OOMHandler() = default;
    
    // 禁止拷贝和移动
    OOMHandler(const OOMHandler&) = delete;
    OOMHandler& operator=(const OOMHandler&) = delete;
    OOMHandler(OOMHandler&&) = delete;
    OOMHandler& operator=(OOMHandler&&) = delete;
    
    std::mutex oomStackMutex_;  // 保护OOM调用栈队列的互斥锁
    std::queue<CallStackString> oomStackQueue_;  // 存储OOM调用栈的队列
};

} // namespace MemScope

#endif // OOM_HANDLER_H