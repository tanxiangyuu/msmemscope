// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef EVENT_REPORT_H
#define EVENT_REPORT_H

#include <memory>
#include <string>
#include <mutex>
#include <cstring>
#include "host_injection/core/LocalProcess.h"
#include "record_info.h"

namespace Leaks {
/*
 * EventReport类主要功能：
 * 1. 将劫持记录的信息传回到工具进程
*/
class EventReport {
public:
    static EventReport& Instance(CommType type);
    bool ReportMalloc(uint64_t addr, uint64_t size, MemOpSpace space, unsigned long long flag);
    bool ReportFree(uint64_t addr);
    bool ReportKernelLaunch(KernelLaunchRecord& kernelLaunchRecord);
    bool ReportAclItf(AclOpType aclOpType);
    bool ReportMark(MstxRecord& mstxRecord);
private:
    explicit EventReport(CommType type);
    uint64_t recordIndex_ = 0;
    uint64_t aclItfRecordIndex_ = 0;
    uint64_t kernelLaunchRecordIndex_ = 0;
    std::mutex mutex_;
};

}

#endif