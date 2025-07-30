// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 
#ifndef ANALYZER_BASE_H
#define ANALYZER_BASE_H
 
#include "event.h"
#include "memory_state_manager.h"
 
namespace Leaks {
 
class AnalyzerBase {
public:
    virtual ~AnalyzerBase() = default;  // 虚析构函数，确保正确析构派生类
 
    // 纯虚函数，要求派生类必须实现
    virtual void EventHandle(std::shared_ptr<EventBase>& event, MemoryState* state) = 0;
};
 
}
 
#endif