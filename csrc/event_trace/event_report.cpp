// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "event_report.h"
#include "log.h"
#include "protocol.h"
#include "serializer.h"

namespace Leaks {

MemOpRecord CreateMemRecord(MemOpType type, MemOpSpace space, uint64_t addr, uint64_t size)
{
    auto record = MemOpRecord {};
    record.memType = type;
    record.space = space;
    record.addr = addr;
    record.memSize = size;

    return record;
}

EventReport& EventReport::Instance(void)
{
    static EventReport instance;
    return instance;
}

EventReport::EventReport()
{
    (void)LocalProcess::GetInstance(CommType::SOCKET); // 连接server
    return;
}

bool EventReport::ReportMalloc(uint64_t addr, uint64_t size, MemOpSpace space)
{
    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord {};
    eventRecord.type = RecordType::MEMORY_RECORD;
    eventRecord.record.memoryRecord = CreateMemRecord(MemOpType::MALLOC, space, addr, size);
    std::lock_guard<std::mutex> guard(mutex_);
    eventRecord.record.memoryRecord.recordIndex = ++recordIndex_;
    auto sendNums = LocalProcess::GetInstance(CommType::SOCKET).Notify(Serialize(head, eventRecord));
    Utility::LogInfo("client malloc record, index: %u, addr: 0x%lx, size: %u, space: %u",
        recordIndex_, addr, size, space);
    return (sendNums >= 0);
}

bool EventReport::ReportFree(uint64_t addr)
{
    PacketHead head = {PacketType::RECORD};
    auto eventRecord = EventRecord {};
    eventRecord.type = RecordType::MEMORY_RECORD;
    eventRecord.record.memoryRecord = CreateMemRecord(MemOpType::FREE, MemOpSpace::INVALID, addr, 0);
    std::lock_guard<std::mutex> guard(mutex_);
    eventRecord.record.memoryRecord.recordIndex = ++recordIndex_;
    auto sendNums = LocalProcess::GetInstance(CommType::SOCKET).Notify(Serialize(head, eventRecord));
    Utility::LogInfo("client free record, index: %u, addr: 0x%lx", recordIndex_, addr);
    return (sendNums >= 0);
}

}