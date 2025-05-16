// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef RECORD_INFO_H
#define RECORD_INFO_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <sys/types.h>

namespace Leaks {

constexpr int32_t GD_INVALID_NUM = 9999;
const size_t KERNELNAME_MAX_SIZE = 128;
const size_t ATB_NAME_MAX_SIZE = 64;
const size_t ATEN_NAME_MAX_SIZE = 128;
const size_t OP_NAME_MAX_SIZE = 128;
const size_t ATB_PARAMS_MAX_SIZE = 128;
const size_t MEM_ATTR_MAX_SIZE = 128;

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
    int8_t deviceType;          // 20-npu
    int8_t deviceIndex;
    uint8_t dataType;           // 0-malloc, 1-free, 2-block_free
    uint8_t allocatorType;      // 0-Inner(for PTA), 1-external(for GE)
    int64_t ptr;
    int64_t allocSize;
    int64_t totalAllocated;
    int64_t totalReserved;
    int64_t totalActive;
    int64_t streamPtr;
};

struct TorchNpuRecord {
    uint64_t recordIndex;
    uint64_t kernelIndex;       // 当前所属kernellaunch索引
    uint64_t pid;
    uint64_t tid;
    uint64_t timeStamp;
    int32_t devId;
    MemoryUsage memoryUsage;
};

using AtbMemPoolRecord = TorchNpuRecord;

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

enum class DeviceType : uint8_t {
    NPU = 0,
    CPU = 1,
};

enum class PyTraceType : uint8_t {
    PYCALL = 0,
    PYEXCEPTION,
    PYLINE,
    PYRETURN,
    CCALL,
    CEXCEPTION,
    CRETURN,
    PYOPCODE,
};

struct MemOpRecord {
    uint64_t recordIndex;       // 记录索引
    uint64_t kernelIndex;       // 当前所属kernellaunch索引
    unsigned long long flag;    // 内存属性
    int32_t modid;              // moduleID
    uint64_t pid;
    uint64_t tid;
    DeviceType devType;         // 所属device类型，0为npu，1为cpu
    int32_t devId;              // 所属device id
    MemOpType memType;          // 内存操作类型：malloc还是free
    MemOpSpace space;           // 内存操作空间：device还是host
    uint64_t addr;              // 地址
    uint64_t memSize;           // 操作大小
    uint64_t timeStamp;
};

struct AclItfRecord {
    uint64_t pid;
    uint64_t tid;
    int32_t devId;              // 所属device id
    uint64_t recordIndex;       // 记录索引
    uint64_t kernelIndex;       // 当前所属kernellaunch索引
    uint64_t aclItfRecordIndex; // aclItf索引
    AclOpType type;             // 资源申请还是释放
    uint64_t timeStamp;
};

struct KernelLaunchRecord {
    uint64_t pid;
    uint64_t tid;
    int32_t devId;              // 所属device id
    uint64_t kernelLaunchIndex; // kernelLaunch索引
    uint64_t recordIndex;       // 记录索引
    KernelLaunchType type;      // KernelLaunch类型
    uint64_t timeStamp;
    int32_t streamId;           // streamId
    uint32_t blockDim;          // 算子核函数运行所需核数
    char kernelName[KERNELNAME_MAX_SIZE];  // kernel名称
};

enum class MarkType : int32_t {
    MARK_A = 0,
    RANGE_START_A,
    RANGE_END,
};

struct MstxRecord {
    MarkType markType;
    uint64_t timeStamp;
    uint64_t pid;
    uint64_t tid;
    int32_t devId;              // 所属device id
    uint64_t rangeId;           // 只有Range才会存在ID，纯mark默认为0, Rangeid从1开始递增
    int32_t streamId;           // streamId, range end对应的值为-1
    uint64_t stepId;            // 只有Range类型才有stepId, 默认为0, 记录当前step的ID编号，从1开始递增
    char markMessage[256U];
    uint64_t recordIndex;       // 记录索引
    uint64_t kernelIndex;       // 当前所属kernellaunch索引
};

enum class OpEventType : uint8_t {
    ATEN_START = 0,
    ATEN_END,
    ATB_START,
    ATB_END,
};

enum class KernelEventType : uint8_t {
    KERNEL_START = 0,
    KERNEL_END,
};

struct AtbOpExecuteRecord {
    OpEventType eventType;
    int32_t devId;
    uint64_t timestamp;
    uint64_t pid;
    uint64_t tid;
    uint64_t recordIndex;
    char name[ATB_NAME_MAX_SIZE];
    char params[ATB_PARAMS_MAX_SIZE];
};

struct AtbKernelRecord {
    KernelEventType eventType;
    int32_t devId;
    uint64_t timestamp;
    uint64_t pid;
    uint64_t tid;
    uint64_t recordIndex;
    char name[ATB_NAME_MAX_SIZE];
    char params[ATB_PARAMS_MAX_SIZE];
};

struct AtenOpLaunchRecord {
    OpEventType eventType;
    int32_t devId;
    uint64_t timestamp;
    uint64_t pid;
    uint64_t tid;
    uint64_t recordIndex;
    char name[ATEN_NAME_MAX_SIZE];
};

enum class AccessType : uint8_t {
    READ = 0,
    WRITE,
    UNKNOWN,
};

struct MemAccessRecord {
    AccessType eventType;
    int32_t devId;
    uint64_t timestamp;
    uint64_t pid;
    uint64_t tid;
    uint64_t recordIndex;
    DeviceType devType;         // 所属device类型，0为npu，1为cpu
    uint64_t addr;              // 地址
    uint64_t memSize;           // 操作大小
    char name[OP_NAME_MAX_SIZE];
    char attr[MEM_ATTR_MAX_SIZE];
};

enum class RecordType {
    MEMORY_RECORD = 0,
    ACL_ITF_RECORD,
    KERNEL_LAUNCH_RECORD,
    MSTX_MARK_RECORD,
    TORCH_NPU_RECORD,
    ATB_MEMORY_POOL_RECORD,
    ATB_OP_EXECUTE_RECORD,
    ATB_KERNEL_RECORD,
    ATEN_OP_LAUNCH_RECORD,
    MEM_ACCESS_RECORD,
    INVALID_RECORD,
};

// 事件记录载体
struct EventRecord {
    RecordType type;
    union {
        TorchNpuRecord torchNpuRecord;
        MemOpRecord memoryRecord;
        AclItfRecord aclItfRecord;
        KernelLaunchRecord kernelLaunchRecord;
        MstxRecord mstxRecord;
        AtbMemPoolRecord atbMemPoolRecord;
        AtbOpExecuteRecord atbOpExecuteRecord;
        AtbKernelRecord atbKernelRecord;
        AtenOpLaunchRecord atenOpLaunchRecord;
        MemAccessRecord memAccessRecord;
    } record;
    uint64_t pyStackLen;
    uint64_t cStackLen;
    char buffer[0];
    explicit EventRecord(RecordType type) : type(type)
    {}
    EventRecord() = default;
};
struct CallStackInfo {
    uint64_t pyLen;
    uint64_t cLen;
    char *pyStack;
    char *cStack;
};
struct CallStackString {
    std::string pyStack;
    std::string cStack;
    CallStackString()
    {
        pyStack = "\"\"";
        cStack = "\"\"";
    }
    CallStackString(std::string& c, std::string& python)
    {
        pyStack = python == "" ? "\"\"" : python;
        cStack = c == "" ? "\"\"" : c;
    }
};
struct Record {
    EventRecord eventRecord;
    CallStackInfo callStackInfo;
};

}
#endif