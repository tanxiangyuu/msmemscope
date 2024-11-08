// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "analyzer.h"
#include "log.h"

namespace Leaks {

Analyzer::Analyzer(const AnalysisConfig &config)
{
    config_ = config;
}

void Analyzer::Do(const EventRecord &record)
{
    auto memRecord = record.record.memoryRecord;
    if (memRecord.memType == MemOpType::MALLOC) {
        Utility::LogInfo("server malloc record, index: %u, addr: 0x%lx, size: %u, space: %u",
            memRecord.recordIndex, memRecord.addr, memRecord.memSize, memRecord.space);
    } else if (memRecord.memType == MemOpType::FREE) {
        Utility::LogInfo("server free record, index: %u, addr: 0x%lx", memRecord.recordIndex, memRecord.addr);
    }

    return;
}

}