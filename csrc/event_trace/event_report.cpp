// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "event_report.h"

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

}