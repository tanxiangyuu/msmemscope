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

#ifndef ANALYZER_BASE_H
#define ANALYZER_BASE_H

#include <vector>

#include "event.h"
#include "memory_state_manager.h"

namespace MemScope
{

class AnalyzerBase
{
   public:
    virtual ~AnalyzerBase() = default;  // 虚析构函数，确保正确析构派生类

    // 纯虚函数，要求派生类必须实现
    virtual void EventHandle(std::shared_ptr<EventBase>& event, MemoryState* state) = 0;
    // 新增:分析器名字(控制通道display analyzer用;所有经EventDispatcher::Subscribe
    // 注册的类严格继承AnalyzerBase并实现GetName,名字为display analyzer枚举来源)
    virtual const char* GetName() const = 0;
    // 新增:控制面能力声明(verb=display/set, key=数据或配置键,如 {"display","memory summary"});
    // 默认空=该分析器暂无控制面命令;各分析器能力集覆盖与display analyzer明细展示在后续迭代补齐
    struct ControlCapability
    {
        const char* verb;
        const char* key;
    };
    virtual std::vector<ControlCapability> GetControlCapabilities() const { return {}; }
};

}  // namespace MemScope

#endif
