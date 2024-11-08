// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef EVENT_REPORT_H
#define EVENT_REPORT_H

#include <memory>
#include <string>
#include <mutex>
#include "host_injection/core/LocalProcess.h"
#include "record_info.h"

namespace Leaks {
/*
 * EventReport类主要功能：
 * 1. 将劫持记录的信息传回到工具进程
*/
class EventReport {
public:
    static EventReport& Instance(void);
    bool ReportMalloc(uint64_t addr, uint64_t size, MemOpSpace space);
    bool ReportFree(uint64_t addr);
private:
    EventReport();

    uint64_t recordIndex_ = 0;
    std::mutex mutex_;
};

}

#endif