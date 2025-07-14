// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#include "event_trace/kernel_hooks/driver_prof_api.h"
#include "event_trace/kernel_hooks/kernel_event_trace.h"
#include "event_trace/kernel_hooks/runtime_prof_api.h"
#include "event_trace/kernel_hooks/stars_common.h"

using namespace Leaks;

TEST(TestKernelExcuteTrace, TestKernelExcuteTraceNormal)
{
    uint32_t deviceId = 0;
    uint64_t sysCnt = 0;
    int16_t streamId = 1;
    int16_t taskId = 2;
    StartDriverKernelInfoTrace(static_cast<int32_t>(deviceId));
    RegisterRtProfileCallback();

    uint32_t agingFlag = 1;
    int value = 42;
    const void* data = static_cast<const void*>(&value);
    uint32_t length = sizeof(struct MsprofCompactInfo);
    CompactInfoReporterCallbackImpl(agingFlag, data, length);

    GetRealTimeFromSysCnt(deviceId, sysCnt);

    // aclnn下发
    auto taskKey = std::make_tuple(static_cast<int16_t>(deviceId), streamId, taskId);

    std::string hashInfo = "add";
    uint64_t hashId = GetHashIdCallBackImply(hashInfo.c_str(), hashInfo.size());
    RuntimeKernelLinker::GetInstance().RuntimeTaskInfoLaunch(taskKey, hashId);

    std::string name = "add";
    uint64_t startTime = 123;
    uint64_t endTime = 1234;
    // kernel下发
    RuntimeKernelLinker::GetInstance().KernelLaunch();

    // kernel执行
    KernelEventTrace::GetInstance().KernelStartExcute(taskKey, startTime);
    KernelEventTrace::GetInstance().KernelEndExcute(taskKey, endTime);
    EndDriverKernelInfoTrace();
}

TEST(TestKernelExcuteTrace, TestCompactInfoReporterCallbackImpl)
{
    uint32_t agingFlag = 1;
    const void* data = nullptr;
    uint32_t length = sizeof(struct MsprofCompactInfo);
    CompactInfoReporterCallbackImpl(agingFlag, data, length);

    MsprofCompactInfo value;
    value.level = MSPROF_REPORT_RUNTIME_LEVEL;
    value.type = RT_PROFILE_TYPE_TASK_TRACK;
    const void* data1 = static_cast<const void*>(&value);
    CompactInfoReporterCallbackImpl(agingFlag, data1, length);
}

TEST(TestKernelExcuteTrace, TestGetHashIdCallBackImply)
{
    const char* hashInfo  = nullptr;
    uint32_t length = sizeof(struct MsprofCompactInfo);
    GetHashIdCallBackImply(hashInfo, length);
}

TEST(TestKernelExcuteTrace, TestGetStreamIdFuncAbnormalInput)
{
    uint16_t streamId = STREAM_JUDGE_BIT12_OPERATOR;
    uint16_t taskId = 1;
    GetStreamId(streamId, taskId);
    streamId = STREAM_JUDGE_BIT13_OPERATOR;
    GetStreamId(streamId, taskId);
}

TEST(TestKernelExcuteTrace, TestGetTaskIdFuncAbnormalInput)
{
    uint16_t streamId = STREAM_JUDGE_BIT12_OPERATOR;
    uint16_t taskId = 1;
    GetTaskId(streamId, taskId);
    streamId = STREAM_JUDGE_BIT13_OPERATOR;
    GetTaskId(streamId, taskId);
}