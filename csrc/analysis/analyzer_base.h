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
 
#include "event.h"
#include "memory_state_manager.h"
 
namespace MemScope {
 
class AnalyzerBase {
public:
    virtual ~AnalyzerBase() = default;  // 虚析构函数，确保正确析构派生类
 
    // 纯虚函数，要求派生类必须实现
    virtual void EventHandle(std::shared_ptr<EventBase>& event, MemoryState* state) = 0;
};
 
}
 
#endif