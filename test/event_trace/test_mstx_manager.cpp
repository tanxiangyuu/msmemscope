// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "event_trace/mstx_hooks/mstx_manager.h"

#include <gtest/gtest.h>
using namespace Leaks;

TEST(MstxManagerTest, ReportMarkATest) {
    const char* msg = "Test Message A";
    uint32_t streamId = 0;
    MstxManager::GetInstance().ReportMarkA(msg, streamId);
}

TEST(MstxManagerTest, ReportMarkATest_Nullptr_Msg_Strcpy_Failed) {
    const char* msg = nullptr;
    uint32_t streamId = 0;
    MstxManager::GetInstance().ReportMarkA(msg, streamId);
}

TEST(MstxManagerTest, ReportRangeStartTest) {
    const char* msg = "Test Message A";
    uint32_t streamId = 0;
    std::uint64_t rangeId = MstxManager::GetInstance().ReportRangeStart(msg, streamId);
    EXPECT_GT(rangeId, 0);
}

TEST(MstxManagerTest, ReportRangeStartTest_Nullptr_Msg_Strcpy_Failed) {
    const char* msg = nullptr;
    uint32_t streamId = 0;
    std::uint64_t rangeId = MstxManager::GetInstance().ReportRangeStart(msg, streamId);
    EXPECT_GT(rangeId, 0);
}

TEST(MstxManagerTest, ReportRangeEndTest) {
    MstxManager::GetInstance().ReportRangeEnd(1);
}

TEST(MstxManagerTest, GetRangeIdTest) {
    const char* msg = "GetRangeIdTest";
    uint32_t streamId = 0;
    uint64_t firstId = MstxManager::GetInstance().ReportRangeStart(msg, streamId);
    uint64_t secondId = MstxManager::GetInstance().ReportRangeStart(msg, streamId);
    EXPECT_EQ(secondId, firstId +1);
}

TEST(MstxManagerTest, ReportDomainCreateATest_Get_Nullptr)
{
    char const *domainName = nullptr;
    auto ret = MstxManager::GetInstance().ReportDomainCreateA(domainName);
    ASSERT_EQ(ret, nullptr);
}

TEST(MstxManagerTest, ReportHeapRegisterTest_Get_Nullptr)
{
    mstxDomainHandle_t domain = nullptr;
    mstxMemHeapDesc_t const *desc = nullptr;
    auto ret = MstxManager::GetInstance().ReportHeapRegister(domain, desc);
    ASSERT_EQ(ret, nullptr);
}

TEST(MstxManagerTest, ReportRegionsRegisterTest_Get_Nullptr)
{
    mstxDomainHandle_t domain = nullptr;
    mstxMemRegionsRegisterBatch_t const *desc = nullptr;
    MstxManager::GetInstance().ReportRegionsRegister(domain, desc);
}

TEST(MstxManagerTest, ReportRegionsUnregisterTest_Get_Nullptr)
{
    mstxDomainHandle_t domain = nullptr;
    mstxMemRegionsUnregisterBatch_t const *desc = nullptr;
    MstxManager::GetInstance().ReportRegionsUnregister(domain, desc);
}