// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "event_report.h"
#include <chrono>
#include "log.h"
#include "protocol.h"
#include "serializer.h"
#include "utils.h"
#include "vallina_symbol.h"

namespace Leaks {


constexpr uint64_t MEM_MODULE_ID_BIT = 56;
constexpr uint64_t MEM_VIRT_BIT = 10;
constexpr uint64_t MEM_SVM_VAL = 0x0;
constexpr uint64_t MEM_DEV_VAL = 0x1;
constexpr uint64_t MEM_HOST_VAL = 0x2;
constexpr uint64_t MEM_DVPP_VAL = 0x3;
MemOpSpace GetMemOpSpace(unsigned long long flag)
{
    // bit10~13: virt mem type(svm\dev\host\dvpp)
    int32_t memType = (flag & 0b11110000000000) >> MEM_VIRT_BIT;
    MemOpSpace space = MemOpSpace::INVALID;
    switch (memType) {
        case MEM_SVM_VAL:
            space = MemOpSpace::SVM;
            break;
        case MEM_DEV_VAL:
            space = MemOpSpace::DEVICE;
            break;
        case MEM_HOST_VAL:
            space = MemOpSpace::HOST;
            break;
        case MEM_DVPP_VAL:
            space = MemOpSpace::DVPP;
            break;
        default:
            Utility::LogError("No matching memType for %d .", memType);
    }
    return space;
}
inline int32_t GetMallocModuleId(unsigned long long flag)
{
    // bit56~63: model id
    return (flag & 0xFF00000000000000) >> MEM_MODULE_ID_BIT;
}

constexpr unsigned long long FLAG_INVALID = UINT64_MAX;

constexpr int32_t GD_INVALID_NUM = 9999;

constexpr int32_t INVALID_MODID = -1;

RTS_API rtError_t GetDevice(int32_t *devid)
{
    char const *sym = "rtGetDevice";
    using RtGetDevice = decltype(&GetDevice);
    auto vallina = VallinaSymbol<RuntimeLibLoader>::Instance().Get<RtGetDevice>(sym);
    if (vallina == nullptr) {
        Utility::LogError("vallina func get FAILED");
        return RT_ERROR_RESERVED;
    }
    rtError_t ret = vallina(devid);
    return ret;
}
MemOpRecord CreateMemRecord(MemOpType type, unsigned long long flag, MemOpSpace space, uint64_t addr, uint64_t size)
{
    MemOpRecord record;
    record.flag = flag;
    record.memType = type;
    record.space = space;
    record.addr = addr;
    record.memSize = size;
    record.pid = Utility::GetPid();
    record.tid = Utility::GetTid();
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    record.timeStamp = time;
    return record;
}

AclItfRecord CreateAclItfRecord(AclOpType type)
{
    auto record = AclItfRecord {};
    record.type = type;
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    record.pid = Utility::GetPid();
    record.tid = Utility::GetTid();
    record.timeStamp = time;
    return record;
}

KernelLaunchRecord CreateKernelLaunchRecord(KernelLaunchRecord kernelLaunchRecord)
{
    auto record = KernelLaunchRecord {};
    record = kernelLaunchRecord;
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    record.timeStamp = time;
    record.pid = Utility::GetPid();
    record.tid = Utility::GetTid();
    return record;
}

EventReport& EventReport::Instance(CommType type)
{
    static EventReport instance(type);
    return instance;
}

EventReport::EventReport(CommType type)
{
    (void)LocalProcess::GetInstance(type); // 连接server
    return;
}
bool EventReport::ReportTorchNpu(TorchNpuRecord &torchNpuRecord)
{
    PacketHead head = {PacketType::RECORD};
    EventRecord eventrecord;
    eventrecord.type = RecordType::TORCH_NPU_RECORD;
    eventrecord.record.torchNpuRecord = torchNpuRecord;
    std::lock_guard<std::mutex> guard(mutex_);
    eventrecord.record.torchNpuRecord.recordIndex = ++recordIndex_;
    auto sendNums = LocalProcess::GetInstance(CommType::SOCKET).Notify(Serialize(head, eventrecord));
    Utility::LogInfo("TorchNpu Record, index: %u", recordIndex_);
    return (sendNums >= 0);
}
bool EventReport::ReportMalloc(uint64_t addr, uint64_t size, unsigned long long flag)
{
    int32_t devid = GD_INVALID_NUM;
    if (GetDevice(&devid) == RT_ERROR_INVALID_VALUE || devid == GD_INVALID_NUM) {
        Utility::LogError("RT_ERROR_INVALID_VALUE, %d!!!!!!!!!!!!", devid);
    }
    int32_t moduleId = GetMallocModuleId(flag);
    MemOpSpace space = GetMemOpSpace(flag);
    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord {};
    eventRecord.type = RecordType::MEMORY_RECORD;
    eventRecord.record.memoryRecord = CreateMemRecord(MemOpType::MALLOC, flag, space, addr, size);
    eventRecord.record.memoryRecord.devid = devid;
    eventRecord.record.memoryRecord.modid = moduleId;
    std::lock_guard<std::mutex> guard(mutex_);
    eventRecord.record.memoryRecord.recordIndex = ++recordIndex_;
    eventRecord.record.memoryRecord.kernelIndex = kernelLaunchRecordIndex_;
    auto sendNums = LocalProcess::GetInstance(CommType::SOCKET).Notify(Serialize(head, eventRecord));
    Utility::LogInfo("client malloc record, index: %u, addr: 0x%lx, size: %u, space: %u, flag: %llu",
        recordIndex_, addr, size, space, flag);
    return (sendNums >= 0);
}

bool EventReport::ReportFree(uint64_t addr)
{
    int32_t devid = GD_INVALID_NUM;
    if (GetDevice(&devid) == RT_ERROR_INVALID_VALUE || devid == GD_INVALID_NUM) {
        Utility::LogError("RT_ERROR_INVALID_VALUE, %d!!!!!!!!!!!!", devid);
    }
    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord {};
    eventRecord.type = RecordType::MEMORY_RECORD;
    eventRecord.record.memoryRecord = CreateMemRecord(MemOpType::FREE, FLAG_INVALID, MemOpSpace::INVALID, addr, 0);
    eventRecord.record.memoryRecord.devid = devid;
    eventRecord.record.memoryRecord.modid = INVALID_MODID;
    std::lock_guard<std::mutex> guard(mutex_);
    eventRecord.record.memoryRecord.recordIndex = ++recordIndex_;
    eventRecord.record.memoryRecord.kernelIndex = kernelLaunchRecordIndex_;
    auto sendNums = LocalProcess::GetInstance(CommType::SOCKET).Notify(Serialize(head, eventRecord));
    Utility::LogInfo("client free record, index: %u, addr: 0x%lx", recordIndex_, addr);
    return (sendNums >= 0);
}

bool EventReport::ReportMark(MstxRecord& mstxRecord)
{
    if (mstxRecord.streamId == -1) { // range end打点无需输出streamId信息，通过rangeId与start匹配
        Utility::LogInfo("this mark point message is %s", mstxRecord.markMessage);
    } else {
        Utility::LogInfo("this mark point message is %s, streamId is %d", mstxRecord.markMessage, mstxRecord.streamId);
    }
    Utility::LogInfo("this mark point id is %llu", mstxRecord.rangeId);
    return true;
}

bool EventReport::ReportKernelLaunch(KernelLaunchRecord& kernelLaunchRecord)
{
    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord{};
    eventRecord.type = RecordType::KERNEL_LAUNCH_RECORD;
    eventRecord.record.kernelLaunchRecord = CreateKernelLaunchRecord(kernelLaunchRecord);
    std::lock_guard<std::mutex> guard(mutex_);
    eventRecord.record.kernelLaunchRecord.kernelLaunchIndex = ++kernelLaunchRecordIndex_;
    eventRecord.record.kernelLaunchRecord.recordIndex = ++recordIndex_;
    auto sendNums = LocalProcess::GetInstance(CommType::SOCKET).Notify(Serialize(head, eventRecord));
    Utility::LogInfo("client kernelLaunch record, name: %s, index: %u, type: %u, time: %u, streamId: %d, blockDim: %u",
        eventRecord.record.kernelLaunchRecord.kernelName,
        kernelLaunchRecordIndex_,
        eventRecord.record.kernelLaunchRecord.type,
        eventRecord.record.kernelLaunchRecord.timeStamp,
        eventRecord.record.kernelLaunchRecord.streamId,
        eventRecord.record.kernelLaunchRecord.blockDim);
    return (sendNums >= 0);
}

bool EventReport::ReportAclItf(AclOpType aclOpType)
{
    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord{};
    eventRecord.type = RecordType::ACL_ITF_RECORD;
    eventRecord.record.aclItfRecord = CreateAclItfRecord(aclOpType);
    std::lock_guard<std::mutex> guard(mutex_);
    eventRecord.record.aclItfRecord.recordIndex = ++recordIndex_;
    eventRecord.record.aclItfRecord.aclItfRecord = ++aclItfRecordIndex_;
    auto sendNums = LocalProcess::GetInstance(CommType::SOCKET).Notify(Serialize(head, eventRecord));
    Utility::LogInfo("client aclItf record, index: %u, type: %u, time: %u",
        aclItfRecordIndex_,
        aclOpType,
        eventRecord.record.aclItfRecord.timeStamp);
    return (sendNums >= 0);
}
}