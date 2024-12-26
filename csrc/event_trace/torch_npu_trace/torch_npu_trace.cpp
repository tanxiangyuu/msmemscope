// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "torch_npu_trace.h"

namespace Leaks {

    bool ReportTorchNpuMemData(MemoryUsage memoryUsage, uint64_t pid, uint64_t tid)
    {
        TorchNpuRecord torchNpuRecord;
        torchNpuRecord.memoryUsage = memoryUsage;
        torchNpuRecord.pid = pid;
        torchNpuRecord.tid = tid;
        if (!EventReport::Instance(CommType::SOCKET).ReportTorchNpu(torchNpuRecord)) {
            return false;
        }
        return true;
    }
}