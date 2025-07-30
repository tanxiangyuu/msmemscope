// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef MEMORY_STATE_MANAGER_H
#define MEMORY_STATE_MANAGER_H

#include <unordered_map>
#include <string>
#include <vector>

#include "state_manager.h"
#include "event.h"

namespace Leaks {

class MemoryStateKey : StateKey {
public:
    uint64_t pid;
    uint64_t addr;

    MemoryStateKey(uint64_t pid, uint64_t addr) : pid(pid), addr(addr) {}

    // 必须实现相等运算符
    bool operator==(const MemoryStateKey& other) const
    {
        return (pid == other.pid) && (addr == other.addr);
    }
};

struct MemoryStateKeyHasher {
    std::size_t operator()(const MemoryStateKey& key) const
    {
        size_t pidHash = std::hash<uint64_t>()(key.pid);
        size_t addrHash = std::hash<uint64_t>()(key.addr);
        return pidHash ^ (addrHash << 1);
    }
};

class MemoryState : public StateBase {
public:
    std::vector<std::shared_ptr<MemoryEvent>> events;
    uint64_t size = 0;
    std::string leaksDefinedOwner;
    std::string userDefinedOwner;
    std::string inefficientType;

    explicit MemoryState() {}

    explicit MemoryState(std::shared_ptr<MemoryEvent>& event)
    {
        events.push_back(event);
        size = event->size;
        leaksDefinedOwner = "";
        userDefinedOwner = event->describeOwner;
        inefficientType = "";
    }
};

class Pool {
public:
    std::unordered_map<MemoryStateKey, MemoryState, MemoryStateKeyHasher> statesMap;

    Pool() {}
};

class MemoryStateManager : StateManager {
public:
    static MemoryStateManager& GetInstance();

    bool AddEvent(std::shared_ptr<MemoryEvent>& event);
    bool DeteleState(const PoolType& poolType, const MemoryStateKey& key);
    Pool* GetStatePool(const PoolType& poolType);
    MemoryState* GetState(const PoolType& poolType, const MemoryStateKey& key);
    MemoryState* FindStateInPool(const PoolType& poolType, const MemoryStateKey& key, uint64_t size);
    std::vector<std::pair<PoolType, MemoryStateKey>> GetAllStateKeys();
private:
    std::unordered_map<PoolType, Pool> poolsMap_;
};

}

#endif