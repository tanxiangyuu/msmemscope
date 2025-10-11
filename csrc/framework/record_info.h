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
    MSTX_MARK_RECORD,
    KERNEL_LAUNCH_RECORD,
    KERNEL_EXCUTE_RECORD,
    OP_LAUNCH_RECORD,
    
    PTA_CACHING_POOL_RECORD,
    PTA_WORKSPACE_POOL_RECORD,
    ATB_MEMORY_POOL_RECORD,
    MINDSPORE_NPU_RECORD,
    MEMORY_POOL_RECORD,

    ATB_OP_EXECUTE_RECORD,
    ATB_KERNEL_RECORD,
    ATEN_OP_LAUNCH_RECORD,
    MEM_ACCESS_RECORD,
    ADDR_INFO_RECORD,
    TRACE_STATUS_RECORD,
    INVALID_RECORD,
};

enum class RecordSubType {
    MALLOC = 0,
    FREE,
    USER_DEFINED,
    PTA_OPTIMIZER_STEP,
    INIT,
    FINALIZE,
    NORMAL,
    HANDLEV2,
    FLAGV2,
    KERNEL_START,
    KERNEL_END,
    ATEN_START,
    ATEN_END,
    ATB_START,
    ATB_END,
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

/**
 * @brief 从 Record 对象中获取指定类型的 TLV 块指针。
 *
 * 本函数用于在一个基于 TLV（Type-Length-Value）布局的 Record 对象中，
 * 按顺序查找第一个匹配指定类型的 TLV 块。TLV 块存储在 Record 尾部，
 * 每个块由类型字段、长度字段和内容部分组成，通常紧跟在 Record 本体之后。
 *
 * 示例用法：
 *   const TLVBlock* tlv = GetTlvBlock(RecordClass, TLVBlockType);
 * @tparam RecordClass 要解析的具体 Record 类型
 * @param record 引用或指针，表示包含 TLV 数据的 Record 实例。
 * @param type TLV 块的类型标识符，用于查找对应数据块。
 * @return 指向匹配 TLV 块起始地址的指针；如果未找到则返回 nullptr。
 *
 * @note
 * - Record 对象必须包含合法的 TLV 数据区域。
 * - 如果存在多个相同类型的 TLV 块，仅返回第一个。
 * - 返回的指针指向整个 TLV 块的起始位置（即 type 字段）。
 */
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

    /**
    * @brief 构造一个 RecordBuffer 对象，用于client->server通信，支持通过参数包传入多组 type-value 键值对。
    *
    * 本函数是一个模板，用于构造 Record 类型的对象，其中每组参数应包括一个类型标识符（type）
    * 和一个对应的值（value），可传入任意数量的 type-value 对。
    *
    * 示例用法：
    *   RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<RecordClass>(TLVBlockType, vlaue);
    *
    * @tparam RecordClass 要构造的具体 Record 类型
    * @param args 参数包，形式为 Type, Value, Type, Value...，数量应为偶数。
    * @return 用于通信的RecordBuffer
    */
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
        if (memcpy_s(data, value.length() + 1, value.c_str(), value.length()) != EOK) {
            std::cout << "[ERROR] make record buffer memcpy failed!" << std::endl;
        }
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
    uint64_t ptr;
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


struct AddrInfo : public RecordBase {
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

struct MemOpRecord : public RecordBase {
    uint64_t kernelIndex;       // 当前所属kernellaunch索引
    unsigned long long flag;    // 内存属性
    int32_t modid;              // moduleID
    MemOpSpace space;           // 内存操作空间：device还是host
    uint64_t addr;              // 地址
    uint64_t memSize;           // 操作大小
    /* TLVBlockType::MEM_OWNER */
};

struct AclItfRecord : public RecordBase {
    uint64_t kernelIndex;       // 当前所属kernellaunch索引
    uint64_t aclItfRecordIndex; // aclItf索引
};

struct TraceStatusRecord : public RecordBase {
    uint8_t status;
};

struct KernelLaunchRecord : public RecordBase {
    int16_t streamId;           // streamId
    int16_t taskId;
    uint64_t kernelLaunchIndex; // kernelLaunch索引
    uint32_t blockDim;          // 算子核函数运行所需核数
    /* TLVBlockType::KERNEL_NAME */
};

struct KernelExcuteRecord : public RecordBase {
    int16_t streamId;
    int16_t taskId;
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
    /* TLVBlockType::ATB_NAME */
    /* TLVBlockType::ATB_PARAMS */
};

struct AtbKernelRecord : public RecordBase {
    /* TLVBlockType::ATB_NAME */
    /* TLVBlockType::ATB_PARAMS */
};

struct AtenOpLaunchRecord : public RecordBase {
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
        TraceStatusRecord traceStatusRecord;
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