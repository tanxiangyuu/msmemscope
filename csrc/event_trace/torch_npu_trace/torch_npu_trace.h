// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef TORCH_NPU_TRACE_H
#define TORCH_NPU_TRACE_H

#include <string>
#include <vector>
#include <iostream>
#include "host_injection/core/LocalProcess.h"
#include "serializer.h"
#include "protocol.h"
#include "record_info.h"
#include "event_report.h"

namespace Leaks {
extern "C" {
bool ReportTorchNpuMemData(MemoryUsage memoryUsage, uint64_t pid, uint64_t tid);
}
}

#endif