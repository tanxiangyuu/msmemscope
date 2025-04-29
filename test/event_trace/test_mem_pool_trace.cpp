// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#define private public
#include "event_trace/memory_pool_trace/memory_pool_trace_manager.h"
#include "event_trace/mstx_hooks/mstx_inject.h"
#include "event_trace/memory_pool_trace/atb_memory_pool_trace.h"
#undef private
#include <gtest/gtest.h>

using namespace Leaks;

TEST(MemPoolTraceTest, MemPoolTraceTestATBHeapRegisterAndRegionRegister)
{
    auto domainHandle = MstxDomainCreateAFunc("atb");
    EXPECT_EQ(domainHandle, ATBMemoryPoolTrace::GetInstance().atbDomain_);

    mstxMemHeapDesc_t heapDesc;
    uint32_t deviceId = 1;
    void const* ptrHeap = reinterpret_cast<void const*>(123);
    mstxMemVirtualRangeDesc_t memRangeDesc{deviceId, ptrHeap, 500};
    heapDesc.typeSpecificDesc = reinterpret_cast<void const*>(&memRangeDesc);
    auto HeapHandle = MstxMemHeapRegisterFunc(domainHandle, &heapDesc);
    EXPECT_EQ(ATBMemoryPoolTrace::GetInstance().memUsageMp_[deviceId].totalReserved, 500);

    void const* ptr = reinterpret_cast<void const*>(123);
    memRangeDesc = {deviceId, ptr, 50};
    mstxMemRegionsRegisterBatch_t desc;
    desc.regionCount = 1;
    desc.regionDescArray = reinterpret_cast<const void *>(&memRangeDesc);
    auto mstxMemRegionHandle = mstxMemRegionHandle_t {};
    auto handleArrayOut = &mstxMemRegionHandle;
    desc.regionHandleArrayOut = handleArrayOut;
    MstxMemRegionsRegisterFunc(domainHandle, &desc);
    EXPECT_EQ(ATBMemoryPoolTrace::GetInstance().memUsageMp_[deviceId].totalAllocated, 50);

    mstxMemRegionsUnregisterBatch_t unregisterBatch;
    unregisterBatch.refCount = 1;
    auto mstxMemRegionRef = new mstxMemRegionRef_t {};

    mstxMemRegionRef->handle = mstxMemRegionHandle;
    unregisterBatch.refArray = mstxMemRegionRef;

    MstxMemRegionsUnregisterFunc(domainHandle, &unregisterBatch);
    EXPECT_EQ(ATBMemoryPoolTrace::GetInstance().memUsageMp_[deviceId].totalAllocated, 0);

    MstxMemHeapUnregisterFunc(domainHandle, HeapHandle);
    EXPECT_EQ(ATBMemoryPoolTrace::GetInstance().memUsageMp_[deviceId].totalReserved, 0);
}