// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef RECORD_INFO_H
#define RECORD_INFO_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <sys/types.h>
#include <memory>

#include "securec.h"
#include "utils.h"

namespace Leaks {

constexpr int32_t GD_INVALID_NUM = 9999;
const size_t KERNELNAME_MAX_SIZE = 128;
const size_t ATB_NAME_MAX_SIZE = 64;
const size_t ATEN_NAME_MAX_SIZE = 128;
const size_t OP_NAME_MAX_SIZE = 128;
const size_t ATB_PARAMS_MAX_SIZE = 128;
const size_t MEM_ATTR_MAX_SIZE = 128;
const size_t ADDR_OWNER_SIZE = 64;

enum class RecordType {
    MEMORY_RECORD = 0,
    ACL_ITF_RECORD,
    KERNEL_LAUNCH_RECORD,
    KERNEL_EXCUTE_RECORD,
    MSTX_MARK_RECORD,
    TORCH_NPU_RECORD,
    ATB_MEMORY_POOL_RECORD,
    MINDSPORE_NPU_RECORD,
    ATB_OP_EXECUTE_RECORD,
    ATB_KERNEL_RECORD,
    ATEN_OP_LAUNCH_RECORD,
    MEM_ACCESS_RECORD,
    ADDR_INFO_RECORD,
    INVALID_RECORD,
};

enum class RecordSubType {
    MALLOC = 0,
    FREE,
};

enum class TLVBlockType : int {
    SKIP,
    CALL_STACK_C,
    CALL_STACK_PYTHON,
    MEM_OWNER,
    KERNEL_NAME,
    ATB_NAME,
    ATB_PARAMS,
    ATEN_NAME,
    OP_NAME,
    MEM_ATTR,
    ADDR_OWNER,
    MARK_MESSAGE,
};

struct TLVBlock {
    TLVBlockType type;
    uint32_t len;
    char data[0];
};

struct RecordBase {
    /* 注意！！！此处length必须放在结构体开头，不要移动!!! */
    uint32_t length;
    RecordType type;
    RecordSubType subtype;
    int32_t devId;
    uint64_t recordIndex;
    uint64_t timestamp;
    uint64_t pid;
    uint64_t tid;
};

template <typename RecordClass>
const TLVBlock* GetTlvBlock(const RecordClass& record, TLVBlockType type)
{
    const char* begin = static_cast<const char*>(static_cast<const void*>(&record));
    uint32_t alloffset = sizeof(RecordClass);
    const char* data = nullptr;
    while (alloffset < record.length) {
        if (alloffset + sizeof(TLVBlock) > record.length) {
            return nullptr;
        }
        data = begin + alloffset;
        if (static_cast<const TLVBlock*>(static_cast<const void*>(data))->type == type) {
            return static_cast<const TLVBlock*>(static_cast<const void*>(data));
        }
        alloffset += (sizeof(TLVBlock) + static_cast<const TLVBlock*>(static_cast<const void*>(data))->len);
    }
    return nullptr;
}

class RecordBuffer {
public:
    ~RecordBuffer() = default;

    template <typename RecordClass>
    static RecordBuffer CreateRecordBuffer()
    {
        RecordBuffer v;
        v.MakeRecordBuffer(sizeof(RecordClass));
        return v;
    }

    template <typename RecordClass, typename... Args>
    static RecordBuffer CreateRecordBuffer(Args&&... args)
    {
        RecordBuffer v;
        v.MakeRecordBuffer(sizeof(RecordClass), std::forward<Args>(args)...);
        return v;
    }

    explicit RecordBuffer(std::string&& msg)
    {
        buffer_ = std::move(msg);
    }

    /* 调用者自行保证转换合法 */
    template <typename RecordClass>
    RecordClass* Cast() const
    {
        return static_cast<RecordClass*>(static_cast<void*>(const_cast<char*>(buffer_.c_str())));
    }

    inline const std::string& Get() const
    {
        return buffer_;
    }

    inline uint32_t Size() const {return static_cast<uint32_t>(buffer_.size());}

private:
    RecordBuffer() = default;

    void MakeRecordBuffer(size_t len)
    {
        buffer_.resize(len);

        RecordBase* record = Cast<RecordBase>();
        record->length = len;
        record->timestamp = Utility::GetTimeNanoseconds();
        record->pid = Utility::GetPid();
        record->tid = Utility::GetTid();
    }

    template <typename... Args>
    void MakeRecordBuffer(size_t len, const TLVBlockType& type, const std::string& value, Args&&... args)
    {
        if (type == TLVBlockType::SKIP) {
            return MakeRecordBuffer(len, std::forward<Args>(args)...);
        }
        uint32_t blockLength = sizeof(TLVBlock) + value.length() + 1;
        MakeRecordBuffer(len + blockLength, std::forward<Args>(args)...);
        char* data = const_cast<char*>(buffer_.c_str()) + len;
        *(static_cast<TLVBlockType*>(static_cast<void*>(data))) = type;
        data += sizeof(TLVBlockType);
        *(static_cast<uint32_t*>(static_cast<void*>(data))) = static_cast<uint32_t>(value.length() + 1);
        data += sizeof(uint32_t);
        /* 此处data buffer长度为blockLength，减去T和L后长度为 value.length() + 1；使用memcpy而非strcpy是为了拷贝效率更高 */
        memcpy_s(data, value.length() + 1, value.c_str(), value.length());
        data[value.length()] = '\0';
    }

    std::string buffer_;
};

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

struct MemPoolRecord : public RecordBase {
    uint64_t kernelIndex;       // 当前所属kernellaunch索引
    MemoryUsage memoryUsage;
    /* TLVBlockType::ADDR_OWNER */
};

enum class AddrInfoType : uint8_t {
    USER_DEFINED = 0U,
    PTA_OPTIMIZER_STEP,
};

struct AddrInfo : public RecordBase {
    AddrInfoType addrInfoType;
    uint64_t addr;
    /* TLVBlockType::ADDR_OWNER */
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

struct MemOpRecord : public RecordBase {
    uint64_t kernelIndex;       // 当前所属kernellaunch索引
    unsigned long long flag;    // 内存属性
    int32_t modid;              // moduleID
    DeviceType devType;         // 所属device类型，0为npu，1为cpu
    MemOpSpace space;           // 内存操作空间：device还是host
    uint64_t addr;              // 地址
    uint64_t memSize;           // 操作大小
    /* TLVBlockType::MEM_OWNER */
};

struct AclItfRecord : public RecordBase {
    uint64_t kernelIndex;       // 当前所属kernellaunch索引
    uint64_t aclItfRecordIndex; // aclItf索引
    AclOpType aclOpType;             // 资源申请还是释放
};

struct KernelLaunchRecord : public RecordBase {
    int16_t streamId;           // streamId
    int16_t taskId;
    uint64_t kernelLaunchIndex; // kernelLaunch索引
    KernelLaunchType kernelLaunchType;      // KernelLaunch类型
    uint32_t blockDim;          // 算子核函数运行所需核数
    /* TLVBlockType::KERNEL_NAME */
};

struct KernelExcuteRecord : public RecordBase {
    int16_t streamId;
    int16_t taskId;
    KernelEventType kernelEventType;       // KernelLaunch类型
    /* TLVBlockType::KERNEL_NAME */
};

enum class MarkType : int32_t {
    MARK_A = 0,
    RANGE_START_A,
    RANGE_END,
};

struct MstxRecord : public RecordBase {
    MarkType markType;
    uint64_t rangeId;           // 只有Range才会存在ID，纯mark默认为0, Rangeid从1开始递增
    int32_t streamId;           // streamId, range end对应的值为-1
    uint64_t stepId;            // 只有Range类型才有stepId, 默认为0, 记录当前step的ID编号，从1开始递增
    uint64_t kernelIndex;       // 当前所属kernellaunch索引
    /* TLVBlockType::MARK_MESSAGE */
};

struct AtbOpExecuteRecord : public RecordBase {
    OpEventType eventType;
    /* TLVBlockType::ATB_NAME */
    /* TLVBlockType::ATB_PARAMS */
};

struct AtbKernelRecord : public RecordBase {
    KernelEventType eventType;
    /* TLVBlockType::ATB_NAME */
    /* TLVBlockType::ATB_PARAMS */
};

struct AtenOpLaunchRecord : public RecordBase {
    OpEventType eventType;
    /* TLVBlockType::ATEN_NAME */
};

enum class AccessType : uint8_t {
    READ = 0,
    WRITE,
    UNKNOWN,
};

enum class AccessMemType : uint8_t {
    ATEN = 0,
    ATB,
};

struct MemAccessRecord : public RecordBase {
    AccessType eventType;
    AccessMemType memType;     // 所属的mem类型
    DeviceType devType;         // 所属device类型，0为npu，1为cpu
    uint64_t addr;              // 地址
    uint64_t memSize;           // 操作大小
    /* TLVBlockType::OP_NAME */
    /* TLVBlockType::MEM_ATTR */
};

// 事件记录载体
struct EventRecord {
    RecordType type;
    union {
        MemPoolRecord memPoolRecord;
        MemOpRecord memoryRecord;
        AclItfRecord aclItfRecord;
        KernelLaunchRecord kernelLaunchRecord;
        KernelExcuteRecord kernelExcuteRecord;
        MstxRecord mstxRecord;
        AtbOpExecuteRecord atbOpExecuteRecord;
        AtbKernelRecord atbKernelRecord;
        AtenOpLaunchRecord atenOpLaunchRecord;
        MemAccessRecord memAccessRecord;
        AddrInfo addrInfo;
    } record;
    uint64_t pyStackLen;
    uint64_t cStackLen;
    char buffer[0];
    explicit EventRecord(RecordType type) : type(type), pyStackLen(0), cStackLen(0)
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
    std::string cStack;
    std::string pyStack;
    CallStackString() {}
    CallStackString(std::string& c, std::string& python) : cStack(c), pyStack(python) {}
};

struct Record {
    EventRecord eventRecord;
    CallStackInfo callStackInfo;
};

}
#endif