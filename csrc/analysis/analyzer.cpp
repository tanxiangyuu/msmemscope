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
    switch (record.type) {
        case RecordType::MEMORY_RECORD: {
            auto memRecord = record.record.memoryRecord;
            switch (memRecord.memType) {
                case MemOpType::MALLOC:
                    Utility::LogInfo("server malloc record, index: %u, addr: 0x%lx, size: %u, space: %u",
                        memRecord.recordIndex,
                        memRecord.addr,
                        memRecord.memSize,
                        memRecord.space);
                    break;
                case MemOpType::FREE:
                    Utility::LogInfo(
                        "server free record, index: %u, addr: 0x%lx", memRecord.recordIndex, memRecord.addr);
                    break;
            }
            break;
        }
        case RecordType::KERNEL_LAUNCH_RECORD: {
            auto kernelLaunchRecord = record.record.kernelLaunchRecord;
            Utility::LogInfo("server kernelLaunch record, index: %u, type: %u, time: %u",
                kernelLaunchRecord.recordIndex,
                kernelLaunchRecord.type,
                kernelLaunchRecord.timeStamp);
            break;
        }
        case RecordType::ACL_ITF_RECORD: {
            auto aclItfRecord = record.record.aclItfRecord;
            Utility::LogInfo("server aclItf record, index: %u, type: %u, time: %u",
                aclItfRecord.recordIndex,
                aclItfRecord.type,
                aclItfRecord.timeStamp);
            break;
        }
        default:
            break;
    }

    return;
}
}