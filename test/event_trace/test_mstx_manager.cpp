// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "event_trace/mstx_hooks/mstx_manager.h"

#include <gtest/gtest.h>
using namespace Leaks;

TEST(MstxManagerTest, ReportMarkATest) {
    const char* msg = "Test Message A";
    MstxManager::GetInstance().ReportMarkA(msg);
}

TEST(MstxManagerTest, ReportRangeStartTest) {
    const char* msg = "Test Message A";
    std::uint64_t rangeId = MstxManager::GetInstance().ReportRangeStart(msg);
    EXPECT_GT(rangeId, 0);
}

TEST(MstxManagerTest, ReportRangeEndTest) {
    MstxManager::GetInstance().ReportRangeEnd(1);
}

TEST(MstxManagerTest, GetRangeIdTest) {
    const char* msg = "GetRangeIdTest";
    uint64_t firstId = MstxManager::GetInstance().ReportRangeStart(msg);
    uint64_t secondId = MstxManager::GetInstance().ReportRangeStart(msg);
    EXPECT_EQ(secondId, firstId +1);
}