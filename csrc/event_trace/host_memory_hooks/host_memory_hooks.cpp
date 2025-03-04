// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 
#include <dlfcn.h>
#include <cstdio>
#include "event_report.h"

using namespace Leaks;

// reportInfo为true时开启数据上报，设为false时暂停数据上报
thread_local bool g_reportInfo = true;

extern "C" void* malloc(size_t size)
{
    static void* (*realMalloc)(size_t) = (void* (*)(size_t))dlsym(RTLD_NEXT, "malloc");     // 获取系统malloc函数
    void* ptr = realMalloc(size);
    if (!ptr) {
        return ptr;
    }

    // 只在以下条件均满足时进行report：
    // 1. 在用户使用mstx打点指定的范围内调用malloc/free；
    // 2. 不在各个Report函数中。
    if (!g_isReportHostMem || g_isInReportFunction) {
        return ptr;
    }

    if (g_reportInfo) {
        g_reportInfo = false;
        if (!EventReport::Instance(CommType::SOCKET).ReportHostMalloc(reinterpret_cast<uint64_t>(ptr),
            static_cast<uint64_t>(size))) {
            printf("Report host malloc event failed.\n");
        }
        g_reportInfo = true;
    }
    return ptr;
}

extern "C" void free(void* ptr)
{
    static void (*realFree)(void*) = (void (*)(void*))dlsym(RTLD_NEXT, "free");             // 获取系统free函数
    realFree(ptr);
    if (!ptr) {
        return;
    }

    if (!g_isReportHostMem || g_isInReportFunction) {
        return;
    }

    if (g_reportInfo) {
        g_reportInfo = false;
        if (!EventReport::Instance(CommType::SOCKET).ReportHostFree(reinterpret_cast<uint64_t>(ptr))) {
            printf("Report host free event failed.\n");
        }
        g_reportInfo = true;
    }
    return;
}