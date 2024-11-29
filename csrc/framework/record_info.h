// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef RECORD_INFO_H
#define RECORD_INFO_H

#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>

namespace Leaks {

enum class MemOpType : uint8_t {
    MALLOC = 0U,
    FREE,
};

enum class MemOpSpace : uint8_t {
    SVM = 0U,
    DEVICE,
    HOST,
    DVPP,
    INVALID,
};

enum class StepType : uint8_t {
    START = 0U,
    STOP,
};

enum class AclOpType : uint8_t {
    INIT = 0U,
    FINALIZE,
};

enum class KernelLaunchType : uint8_t {
    NORMAL = 0U,
    HANDLEV2,
    FLAGV2,
};

struct MemOpRecord {
    uint64_t recordIndex; // 记录索引
    MemOpType memType; // 内存操作类型：malloc还是free
    MemOpSpace space; // 内存操作空间：device还是host
    uint64_t addr; // 地址
    uint64_t memSize; // 操作大小
    uint64_t timeStamp; // 时间戳
};

struct StepRecord {
    uint64_t recordIndex; // 记录索引
    StepType type; // 起始还是终止
    uint64_t timeStamp; // 时间戳
};

struct AclItfRecord {
    uint64_t recordIndex; // 记录索引
    AclOpType type; // 资源申请还是释放
    uint64_t timeStamp; // 时间戳
};

struct KernelLaunchRecord {
    uint64_t recordIndex; // 记录索引
    KernelLaunchType type; // KernelLaunch类型
    uint64_t timeStamp; // 时间戳
};

enum class MarkType : int32_t {
    MARK_A = 0,
    RANGE_START_A,
    RANGE_END,
};

struct MstxRecord {
    MarkType markType;
    uint64_t rangeId; // 只有Range才会存在ID，纯mark默认为0
    char markMessage[64U];
};

enum class RecordType {
    MEMORY_RECORD = 0,
    STEP_RECORD,
    ACL_ITF_RECORD,
    KERNEL_LAUNCH_RECORD,
    MSTX_MARK_RECORD,
};

// 事件记录载体
struct EventRecord {
    RecordType type;
    unsigned long long flag;
    union {
        MemOpRecord memoryRecord;
        StepRecord stepRecord;
        AclItfRecord aclItfRecord;
        KernelLaunchRecord kernelLaunchRecord;
        MstxRecord mstxRecord;
    } record;
};

}
#endif