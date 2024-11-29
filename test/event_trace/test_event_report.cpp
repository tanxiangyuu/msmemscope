// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "event_trace/event_report.h"

#include <gtest/gtest.h>

using namespace Leaks;

TEST(EventReportTest, EventReportInstanceTest) {
    EventReport& instance1 = EventReport::Instance(CommType::MEMORY);
    EventReport& instance2 = EventReport::Instance(CommType::MEMORY);
    EXPECT_EQ(&instance1, &instance2);
}

TEST(EventReportTest, ReportMallocTestDEVICE) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    uint64_t testAddr = 0x12345678;
    uint64_t testSize = 1024;
    unsigned long long testFlag = 2377900603261207558;
    MemOpSpace space = MemOpSpace::DEVICE;
    EXPECT_TRUE(instance.ReportMalloc(testAddr, testSize, space, testFlag));
}

TEST(EventReportTest, ReportMallocTestHost) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    uint64_t testAddr = 0x12345678;
    uint64_t testSize = 1024;
    unsigned long long testFlag = 504403158274934784;
    MemOpSpace space = MemOpSpace::HOST;
    EXPECT_TRUE(instance.ReportMalloc(testAddr, testSize, space, testFlag));
}

TEST(EventReportTest, ReportFreeTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    uint64_t testAddr = 0x12345678;
    EXPECT_TRUE(instance.ReportFree(testAddr));
}


TEST(EventReportTest, ReportMarkTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    MstxRecord record;
    record.rangeId = 123;
    EXPECT_TRUE(instance.ReportMark(record));
}

TEST(EventReportTest, ReportKernelLaunchTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    EXPECT_TRUE(instance.ReportKernelLaunch(KernelLaunchType::NORMAL));
}

TEST(EventReportTest, ReportAclItfTest) {
    EventReport& instance = EventReport::Instance(CommType::MEMORY);
    EXPECT_TRUE(instance.ReportAclItf(AclOpType::INIT));
}