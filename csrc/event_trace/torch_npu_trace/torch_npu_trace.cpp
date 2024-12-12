// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "torch_npu_trace.h"

namespace Leaks {

    bool ReportTorchNpuMemData(MemoryUsage memoryUsage)
    {
        TorchNpuRecord torchNpuRecord;
        torchNpuRecord.memoryUsage = memoryUsage;
        if (!EventReport::Instance(CommType::SOCKET).ReportTorchNpu(torchNpuRecord)) {
            return false;
        }
        return true;
    }
}