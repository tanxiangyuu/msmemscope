// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef RECORD_INFO_H
#define RECORD_INFO_H

#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>

namespace Leaks {

enum class MemoryDataType {
    MEMORY_MALLOC = 0,
    MEMORY_FREE,
    MEMORY_BLOCK_FREE,
    MEMORY_INVALID
};

enum class MemoryAllocatorType {
    ALLOCATOR_INNER = 0,
    ALLOCATOR_EXTERNAL,
    ALLOCATOR_INVALID,
};
struct MemoryUsage {
    int8_t device_type;
    int8_t device_index;
    uint8_t data_type; // MemoryDataType
    uint8_t allocator_type; // MemoryAllocatorType
    int64_t ptr;
    int64_t alloc_size;
    int64_t total_allocated;
    int64_t total_reserved;
    int64_t total_active;
    int64_t stream_ptr;
};
struct TorchNpuRecord {
    MemoryUsage memoryUsage;
};

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
    unsigned long long flag; // flag信息
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
    int32_t streamId; // streamId
    uint32_t blockDim; // 算子核函数运行所需核数
    char kernelName[64U]; // kernel名称
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
    TORCH_NPU_RECORD,
};

// 事件记录载体
struct EventRecord {
    RecordType type;
    union {
        TorchNpuRecord torchNpuRecord;
        MemOpRecord memoryRecord;
        StepRecord stepRecord;
        AclItfRecord aclItfRecord;
        KernelLaunchRecord kernelLaunchRecord;
        MstxRecord mstxRecord;
    } record;
};

}
#endif