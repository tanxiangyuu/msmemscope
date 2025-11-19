// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef STATE_MANAGER_H
#define STATE_MANAGER_H

#include <unordered_map>
#include <memory>

#include "state.h"

namespace MemScope {

enum class PoolType : uint8_t {
    HOST = 0,
    HAL,
    PTA_CACHING,
    PTA_WORKSPACE,
    MINDSPORE,
    ATB,
    INVALID,
};

class StateKey {
public:
    virtual ~StateKey() {};
};

class StateManager {
public:
    virtual ~StateManager() {};
};

}

#endif