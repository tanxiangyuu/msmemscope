// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "torch_npu_trace.h"

namespace Leaks {

    bool GetTorchNpuMemData(MemoryUsage memoryUsage)
    {
        PacketHead head = {PacketType::RECORD};
        EventRecord eventrecord;
        eventrecord.type = RecordType::TORCH_NPU_RECORD;
        TorchNpuRecord torchNpuRecord;
        torchNpuRecord.memoryUsage = memoryUsage;
        eventrecord.record.torchNpuRecord = torchNpuRecord;
        auto sendNums = LocalProcess::GetInstance(CommType::SOCKET).Notify(Serialize(head, eventrecord));
        return (sendNums >= 0);
    }
}

