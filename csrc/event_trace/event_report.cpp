// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "event_report.h"
#include "log.h"

namespace Leaks {

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

bool EventReport::ReportMalloc(uint64_t addr, uint64_t size, uint64_t memInfoSrc)
{
    Utility::LogInfo("Malloc Addr: 0x%lx, size: %u, moduleId: %u", addr, size, memInfoSrc);
    return true;
}

bool EventReport::ReportFree(uint64_t addr)
{
    Utility::LogInfo("Free Addr: 0x%lx", addr);
    return true;
}

}