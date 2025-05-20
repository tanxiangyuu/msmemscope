// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef KERNEL_EVENT_TRACE_H
#define KERNEL_EVENT_TRACE_H

#include <unordered_map>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include "record_info.h"

namespace Leaks {

using TaskKey = std::tuple<int16_t, int16_t, int16_t>; // <deviceId, streamId, taskId> 唯一标识一个算子任务
using ThreadId = uint64_t;

struct AclnnKernelMapInfo {
    TaskKey taskKey;
    std::string kernelName;
};

// 该类关注kernel下发、开始执行和结束执行
// 作为信息采集（hook、驱动上报）和event report之间的桥梁
class KernelEventTrace {
public:
    KernelEventTrace(const KernelEventTrace&) = delete;
    KernelEventTrace& operator=(const KernelEventTrace&) = delete;

    static KernelEventTrace& GetInstance()
    {
        static KernelEventTrace instance;
        return instance;
    }

    void KernelLaunch(KernelLaunchRecord& kernelLaunchRecord, const void *hdl); // kernel下发
    void KernelStartExcute(const TaskKey& key, uint64_t time); // kernel开始执行
    void KernelEndExcute(const TaskKey& key, uint64_t time); // kernel结束执行

    void StartKernelEventTrace();
    void EndKernelEventTrace();
private:
    KernelEventTrace() = default;
    ~KernelEventTrace();
    void CreateReadDataChannel(uint32_t devId);

private:
    std::atomic<bool> started_{true};
    std::thread readTh_;
};

// 用于关联aclnn下发和kernel下发，在一个线程内，aclnn下发和对应的kernel下发是保序的，中间不会插入其他aclnn和kernel下发
class AclnnKernelLaunchMap {
public:
    AclnnKernelLaunchMap(const AclnnKernelLaunchMap&) = delete;
    AclnnKernelLaunchMap& operator=(const AclnnKernelLaunchMap&) = delete;

    static AclnnKernelLaunchMap& GetInstance()
    {
        static AclnnKernelLaunchMap instance;
        return instance;
    }

    void AclnnLaunch(const TaskKey& key);
    void KernelLaunch(std::string &kernelName);

    TaskKey GetTaskKey();
    std::string GetKernelName(const TaskKey& key, KernelEventType type);
private:
    AclnnKernelLaunchMap() = default;
    ~AclnnKernelLaunchMap() = default;

private:
    std::mutex mutex_;
    std::unordered_map<ThreadId, std::vector<AclnnKernelMapInfo>> kernelNameMp_;
};

}

#endif