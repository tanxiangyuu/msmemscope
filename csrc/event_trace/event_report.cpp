// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "event_report.h"
#include <chrono>
#include "log.h"
#include "protocol.h"
#include "serializer.h"

namespace Leaks {

constexpr unsigned long long FLAG_INVALID = UINT64_MAX;

MemOpRecord CreateMemRecord(MemOpType type, unsigned long long flag, MemOpSpace space, uint64_t addr, uint64_t size)
{
    auto record = MemOpRecord {};
    record.flag = flag;
    record.memType = type;
    record.space = space;
    record.addr = addr;
    record.memSize = size;

    return record;
}

AclItfRecord CreateAclItfRecord(AclOpType type)
{
    auto record = AclItfRecord {};
    record.type = type;
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    record.timeStamp = time;
    return record;
}

KernelLaunchRecord CreateKernelLaunchRecord(KernelLaunchType type)
{
    auto record = KernelLaunchRecord {};
    record.type = type;
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    record.timeStamp = time;
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

bool EventReport::ReportMalloc(uint64_t addr, uint64_t size, MemOpSpace space, unsigned long long flag)
{
    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord {};
    eventRecord.type = RecordType::MEMORY_RECORD;
    eventRecord.record.memoryRecord = CreateMemRecord(MemOpType::MALLOC, flag, space, addr, size);
    std::lock_guard<std::mutex> guard(mutex_);
    eventRecord.record.memoryRecord.recordIndex = ++recordIndex_;
    auto sendNums = LocalProcess::GetInstance(CommType::SOCKET).Notify(Serialize(head, eventRecord));
    Utility::LogInfo("client malloc record, index: %u, addr: 0x%lx, size: %u, space: %u, flag: %llu",
        recordIndex_, addr, size, space, flag);
    return (sendNums >= 0);
}

bool EventReport::ReportFree(uint64_t addr)
{
    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord {};
    eventRecord.type = RecordType::MEMORY_RECORD;
    eventRecord.record.memoryRecord = CreateMemRecord(MemOpType::FREE, FLAG_INVALID, MemOpSpace::INVALID, addr, 0);
    std::lock_guard<std::mutex> guard(mutex_);
    eventRecord.record.memoryRecord.recordIndex = ++recordIndex_;
    auto sendNums = LocalProcess::GetInstance(CommType::SOCKET).Notify(Serialize(head, eventRecord));
    Utility::LogInfo("client free record, index: %u, addr: 0x%lx", recordIndex_, addr);
    return (sendNums >= 0);
}

bool EventReport::ReportMark(MstxRecord& mstxRecord)
{
    Utility::LogInfo("this mark point message is %s", mstxRecord.markMessage);
    Utility::LogInfo("this mark point id is %llu", mstxRecord.rangeId);
    return true;
}

bool EventReport::ReportKernelLaunch(KernelLaunchRecord& kernelLaunchRecord)
{
    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord{};
    eventRecord.type = RecordType::KERNEL_LAUNCH_RECORD;
    eventRecord.record.kernelLaunchRecord = kernelLaunchRecord;
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    eventRecord.record.kernelLaunchRecord.timeStamp = time;
    std::lock_guard<std::mutex> guard(mutex_);
    eventRecord.record.kernelLaunchRecord.recordIndex = ++kernelLaunchRecordIndex_;
    auto sendNums = LocalProcess::GetInstance(CommType::SOCKET).Notify(Serialize(head, eventRecord));
    Utility::LogInfo("client kernelLaunch record, name: %s, index: %u, type: %u, time: %u, stream: %d, blockDim: %u",
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
    eventRecord.record.aclItfRecord.recordIndex = ++aclItfRecordIndex_;
    auto sendNums = LocalProcess::GetInstance(CommType::SOCKET).Notify(Serialize(head, eventRecord));
    Utility::LogInfo("client aclItf record, index: %u, type: %u, time: %u",
        aclItfRecordIndex_,
        aclOpType,
        eventRecord.record.aclItfRecord.timeStamp);
    return (sendNums >= 0);
}
}