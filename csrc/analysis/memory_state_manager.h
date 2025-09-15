// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef MEMORY_STATE_MANAGER_H
#define MEMORY_STATE_MANAGER_H

#include <unordered_map>
#include <string>
#include <vector>
#include <mutex>

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
    std::vector<uint64_t> apiId;
    uint64_t size = 0;
    uint64_t allocationId = 0;
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
        std::lock_guard<std::mutex> lock(mtx);
        allocationId = ++count;
    }

    static uint64_t IncrementCount()
    {
        std::lock_guard<std::mutex> lock(mtx);
        ++count;
        return count;
    }

    static void ResetCount()
    {
        std::lock_guard<std::mutex> lock(mtx);
        count = 0;
    }
private:
    static std::mutex mtx;      // 修改count需要加锁
    static uint64_t count;      // static变量，用于分配唯一id
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
    MemoryState* GetState(std::shared_ptr<EventBase>& event);
    MemoryState* GetState(std::shared_ptr<MemoryEvent>& event);
    std::vector<std::pair<PoolType, MemoryStateKey>> GetAllStateKeys();
private:
    MemoryState* FindStateInPool(const PoolType& poolType, const MemoryStateKey& key, uint64_t size);

    std::unordered_map<PoolType, Pool> poolsMap_;
    std::mutex mtx_;
};

}

#endif