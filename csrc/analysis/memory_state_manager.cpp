// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "memory_state_manager.h"

#include "log.h"
#include "utility/utils.h"

namespace Leaks {

MemoryStateManager& MemoryStateManager::GetInstance()
{
    static MemoryStateManager manager{};
    return manager;
}

bool MemoryStateManager::AddEvent(std::shared_ptr<MemoryEvent>& event)
{
    if (event->poolType == PoolType::INVALID) {
        // LOG_DEBUG
        return false;
    }
    if (poolsMap_.find(event->poolType) == poolsMap_.end()) {
        poolsMap_[event->poolType] = Pool{};
    }

    MemoryStateKey key = MemoryStateKey{event->pid, event->addr};
    auto& statesPool = poolsMap_[event->poolType];
    auto& statesMap = statesPool.statesMap;

    // 如果device信息是缺失的，尝试补全
    if (IsInvalidDevice(event->device)
        && statesMap.find(key) != statesMap.end()
        && !statesMap[key].events.empty()) {
        event->device = statesMap[key].events[0]->device;
    }

    // hal和host内存存在free事件没有size信息，在此处匹配到malloc事件并填写size
    if (event->eventType == EventBaseType::FREE
        && (event->poolType == PoolType::HOST || event->poolType == PoolType::HAL)
        && !statesMap[key].events.empty() && statesMap[key].events[0]->eventType == EventBaseType::MALLOC) {
        event->size = statesMap[key].events[0]->size;
    }

    if (event->eventType == EventBaseType::MALLOC) {
        if (statesMap.find(key) == statesMap.end()) {
            statesMap[key] = MemoryState{event};
        } else {
            // 重置相关属性
            // LOG_DEBUG
            statesMap[key].size = event->size;
            statesMap[key].leaksDefinedOwner = "";
            statesMap[key].userDefinedOwner = event->describeOwner;
            statesMap[key].inefficientType = "";
            statesMap[key].events.push_back(event);
        }
    } else {
        auto state = FindStateInPool(event->poolType, key, event->size);
        if (state == nullptr) {
            // 当前事件没有匹配到已有的state，需要新建一个state表示新的内存块
            statesMap[key] = MemoryState{event};
        } else {
            state->events.push_back(event);
        }
    }
    return true;
}

bool MemoryStateManager::DeteleState(const PoolType& poolType, const MemoryStateKey& key)
{
    if (poolsMap_.find(poolType) == poolsMap_.end()) {
        // LOG_DEBUG
        return false;
    }
    auto it = poolsMap_[poolType].statesMap.find(key);
    if (it == poolsMap_[poolType].statesMap.end()) {
        // LOG_DEBUG
        return false;
    }
    poolsMap_[poolType].statesMap.erase(it);
    return true;
}

Pool* MemoryStateManager::GetStatePool(const PoolType& poolType)
{
    if (poolsMap_.find(poolType) == poolsMap_.end()) {
        // LOG_DEBUG
        return nullptr;
    }
    return &(poolsMap_[poolType]);
}

MemoryState* MemoryStateManager::GetState(const PoolType& poolType, const MemoryStateKey& key)
{
    if (poolsMap_.find(poolType) == poolsMap_.end()) {
        // LOG_DEBUG
        return nullptr;
    }
    if (poolsMap_[poolType].statesMap.find(key) == poolsMap_[poolType].statesMap.end()) {
        // LOG_DEBUG
        return nullptr;
    }
    return &(poolsMap_[poolType].statesMap[key]);
}

MemoryState* MemoryStateManager::FindStateInPool(const PoolType& poolType, const MemoryStateKey& key, uint64_t size)
{
    auto statePool = GetStatePool(poolType);
    if ((*statePool).statesMap.find(key) != (*statePool).statesMap.end()) {
        // 直接匹配到相同起始地址
        return &((*statePool).statesMap[key]);
    }

    // 使用的地址空间位于某块已分配的内存内
    uint64_t addr = key.addr;
    for (auto& pair : (*statePool).statesMap) {
        uint64_t startingAddr = pair.first.addr;
        if (addr >= startingAddr
            && Utility::GetAddResult(addr, size) <= Utility::GetAddResult(startingAddr, pair.second.size)) {
            return &(pair.second);
        }
    }

    return nullptr;
}

std::vector<std::pair<PoolType, MemoryStateKey>> MemoryStateManager::GetAllStateKeys()
{
    std::vector<std::pair<PoolType, MemoryStateKey>> result;
    for (auto& poolPair : poolsMap_) {
        for (auto& statePair : poolPair.second.statesMap) {
            result.push_back(std::make_pair(poolPair.first, statePair.first));
        }
    }
    return result;
}
}