// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <sstream>
#define private public
#include "event_trace/memory_pool_trace/memory_pool_trace_manager.h"
#include "event_trace/mstx_hooks/mstx_inject.h"
#include "event_trace/memory_pool_trace/atb_memory_pool_trace.h"
#include "event_trace/memory_pool_trace/mindspore_memory_pool_trace.h"
#include "event_trace/memory_pool_trace/pta_caching_pool_trace.h"
#include "event_trace/memory_pool_trace/pta_workspace_pool_trace.h"

#undef private
#include <gtest/gtest.h>

using namespace MemScope;

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

TEST(MemPoolTraceTest, MemPoolTraceTestAllocateGetNullPtr)
{
    mstxDomainHandle_t domain = nullptr;
    mstxMemHeapDesc_t const *desc;
    EXPECT_EQ(ATBMemoryPoolTrace::GetInstance().Allocate(domain, desc), nullptr);
    EXPECT_EQ(MindsporeMemoryPoolTrace::GetInstance().Allocate(domain, desc), nullptr);
}

TEST(MemPoolTraceTest, MemPoolTraceTestDeallocateGetNullPtr)
{
    mstxDomainHandle_t domain = nullptr;
    mstxMemHeapHandle_t desc;
    ATBMemoryPoolTrace::GetInstance().Deallocate(domain, desc);
    MindsporeMemoryPoolTrace::GetInstance().Deallocate(domain, desc);
}

TEST(MemPoolTraceTest, MemPoolTraceTestReallocateGetNullPtr)
{
    mstxDomainHandle_t domain = nullptr;
    mstxMemRegionsRegisterBatch_t const *desc;
    ATBMemoryPoolTrace::GetInstance().Reallocate(domain, desc);
    MindsporeMemoryPoolTrace::GetInstance().Reallocate(domain, desc);
}

TEST(MemPoolTraceTest, MemPoolTraceTestReleaseGetNullPtr)
{
    mstxDomainHandle_t domain = nullptr;
    mstxMemRegionsUnregisterBatch_t const *desc;
    ATBMemoryPoolTrace::GetInstance().Release(domain, desc);
    MindsporeMemoryPoolTrace::GetInstance().Release(domain, desc);
}

TEST(MemPoolTraceTest, ATBMemoryPoolTraceCreateDomainReturnNull)
{
    EXPECT_EQ(ATBMemoryPoolTrace::GetInstance().CreateDomain("test"), nullptr);
}

TEST(MemPoolTraceTest, MemPoolTraceTestMindsporeHeapRegisterAndRegionRegister)
{
    auto domainHandle = MstxDomainCreateAFunc("mindsporeMemPool");
    EXPECT_EQ(domainHandle, MindsporeMemoryPoolTrace::GetInstance().mindsporeDomain_);

    mstxMemHeapDesc_t heapDesc;
    uint32_t deviceId = 1;
    void const* ptrHeap = reinterpret_cast<void const*>(123);
    mstxMemVirtualRangeDesc_t memRangeDesc{deviceId, ptrHeap, 500};
    heapDesc.typeSpecificDesc = reinterpret_cast<void const*>(&memRangeDesc);
    auto HeapHandle = MstxMemHeapRegisterFunc(domainHandle, &heapDesc);
    EXPECT_EQ(MindsporeMemoryPoolTrace::GetInstance().memUsageMp_[deviceId].totalReserved, 500);

    void const* ptr = reinterpret_cast<void const*>(123);
    memRangeDesc = {deviceId, ptr, 50};
    mstxMemRegionsRegisterBatch_t desc;
    desc.regionCount = 1;
    desc.regionDescArray = reinterpret_cast<const void *>(&memRangeDesc);
    auto mstxMemRegionHandle = mstxMemRegionHandle_t {};
    auto handleArrayOut = &mstxMemRegionHandle;
    desc.regionHandleArrayOut = handleArrayOut;
    MstxMemRegionsRegisterFunc(domainHandle, &desc);
    EXPECT_EQ(MindsporeMemoryPoolTrace::GetInstance().memUsageMp_[deviceId].totalAllocated, 50);

    mstxMemRegionsUnregisterBatch_t unregisterBatch;
    unregisterBatch.refCount = 1;
    auto mstxMemRegionRef = new mstxMemRegionRef_t {};

    mstxMemRegionRef->pointer = ptr;
    unregisterBatch.refArray = mstxMemRegionRef;

    MstxMemRegionsUnregisterFunc(domainHandle, &unregisterBatch);
    EXPECT_EQ(MindsporeMemoryPoolTrace::GetInstance().memUsageMp_[deviceId].totalAllocated, 0);

    MstxMemHeapUnregisterFunc(domainHandle, static_cast<mstxMemHeapHandle_t>(const_cast<void *>(ptrHeap)));
    EXPECT_EQ(MindsporeMemoryPoolTrace::GetInstance().memUsageMp_[deviceId].totalReserved, 0);
}

TEST(MemPoolTraceTest, MemPoolTraceTestMindsporeHeapRegisterAndRegionRegisterReturnNull)
{
    auto domainHandle = MstxDomainCreateAFunc("mindspore");
    EXPECT_NE(domainHandle, MindsporeMemoryPoolTrace::GetInstance().mindsporeDomain_);

    mstxMemHeapDesc_t heapDesc;
    uint32_t deviceId = 1;
    void const* ptrHeap = reinterpret_cast<void const*>(123);
    mstxMemVirtualRangeDesc_t memRangeDesc{deviceId, ptrHeap, 500};
    heapDesc.typeSpecificDesc = reinterpret_cast<void const*>(&memRangeDesc);
    auto HeapHandle = MstxMemHeapRegisterFunc(domainHandle, &heapDesc);
    EXPECT_EQ(MindsporeMemoryPoolTrace::GetInstance().memUsageMp_[deviceId].totalReserved, 0);

    void const* ptr = reinterpret_cast<void const*>(123);
    memRangeDesc = {deviceId, ptr, 50};
    mstxMemRegionsRegisterBatch_t desc;
    desc.regionCount = 1;
    desc.regionDescArray = reinterpret_cast<const void *>(&memRangeDesc);
    auto mstxMemRegionHandle = mstxMemRegionHandle_t {};
    auto handleArrayOut = &mstxMemRegionHandle;
    desc.regionHandleArrayOut = handleArrayOut;
    MstxMemRegionsRegisterFunc(domainHandle, &desc);
    EXPECT_EQ(MindsporeMemoryPoolTrace::GetInstance().memUsageMp_[deviceId].totalAllocated, 0);

    mstxMemRegionsUnregisterBatch_t unregisterBatch;
    unregisterBatch.refCount = 1;
    auto mstxMemRegionRef = new mstxMemRegionRef_t {};

    mstxMemRegionRef->handle = mstxMemRegionHandle;
    unregisterBatch.refArray = mstxMemRegionRef;

    MstxMemRegionsUnregisterFunc(domainHandle, &unregisterBatch);
    EXPECT_EQ(MindsporeMemoryPoolTrace::GetInstance().memUsageMp_[deviceId].totalAllocated, 0);

    MstxMemHeapUnregisterFunc(domainHandle, HeapHandle);
    EXPECT_EQ(MindsporeMemoryPoolTrace::GetInstance().memUsageMp_[deviceId].totalReserved, 0);
}

TEST(MemPoolTraceTest, MindsporeMemoryPoolTraceReleaseRegionHandleMpNull)
{
    mstxDomainRegistration_t domainRegistration{};
    mstxDomainHandle_t  domainHandle = &domainRegistration;
    mstxMemRegionsUnregisterBatch_t desc;
    void const* ptr = reinterpret_cast<void const*>(123);
    mstxMemRegionRef_t memRangeDesc = {mstxMemRegionRefType::MSTX_MEM_REGION_REF_TYPE_HANDLE, ptr};
    desc.refCount = 1;
    desc.refArray = &memRangeDesc;
    MindsporeMemoryPoolTrace::GetInstance().regionHandleMp_ = {};
    MindsporeMemoryPoolTrace::GetInstance().Release(domainHandle, &desc);
}

TEST(MemPoolTraceTest, MemoryPoolTraceManagerRelease)
{
    mstxDomainRegistration_t domainRegistration{};
    mstxDomainHandle_t  domainHandle = &domainRegistration;
    mstxMemRegionsUnregisterBatch_t desc;
    void const* ptr = reinterpret_cast<void const*>(123);
    mstxMemRegionRef_t memRangeDesc = {mstxMemRegionRefType::MSTX_MEM_REGION_REF_TYPE_HANDLE, ptr};
    desc.refCount = 1;
    desc.refArray = &memRangeDesc;
    MemoryPoolTraceManager::GetInstance().Release(domainHandle, &desc);
}

TEST(MemPoolTraceTest, MemoryPoolTraceManagerAllocate)
{
    mstxDomainRegistration_t domainRegistration{};
    mstxDomainHandle_t  domainHandle = &domainRegistration;
    mstxMemHeapDesc_t heapDesc;
    uint32_t deviceId = 1;
    void const* ptrHeap = reinterpret_cast<void const*>(123);
    mstxMemVirtualRangeDesc_t memRangeDesc{deviceId, ptrHeap, 500};
    heapDesc.typeSpecificDesc = reinterpret_cast<void const*>(&memRangeDesc);
    MemoryPoolTraceManager::GetInstance().Allocate(domainHandle, &heapDesc);
}

TEST(MemPoolTraceTest, MemoryPoolTraceManagerDeallocate)
{
    mstxDomainRegistration_t domainRegistration{};
    mstxDomainHandle_t  domainHandle = &domainRegistration;
    mstxMemHeapHandle_t heap;
    MemoryPoolTraceManager::GetInstance().Deallocate(domainHandle, heap);
}

TEST(MemPoolTraceTest, MemoryPoolTraceManagerReallocate)
{
    mstxDomainRegistration_t domainRegistration{};
    mstxDomainHandle_t  domainHandle = &domainRegistration;
    void const* ptr = reinterpret_cast<void const*>(123);
    mstxMemVirtualRangeDesc_t memRangeDesc = {1, ptr, 50};
    mstxMemRegionsRegisterBatch_t desc;
    desc.regionCount = 1;
    desc.regionDescArray = reinterpret_cast<const void *>(&memRangeDesc);
    MemoryPoolTraceManager::GetInstance().Reallocate(domainHandle, &desc);
}

TEST(MemPoolTraceTest, MindsporeMemoryPoolTraceCreateDomainReturnNull)
{
    EXPECT_EQ(MindsporeMemoryPoolTrace::GetInstance().CreateDomain("mdspe"), nullptr);
}

TEST(MemPoolTraceTest, PTACachingPoolTraceTestPtaHeapRegisterAndRegionRegister)
{
    auto domainHandle = MstxDomainCreateAFunc("ptaCaching");
    EXPECT_EQ(domainHandle, PTACachingPoolTrace::GetInstance().ptaCachingDomain_);
 
    mstxMemHeapDesc_t heapDesc;
    uint32_t deviceId = 1;
    void const* ptrHeap = reinterpret_cast<void const*>(123);
    mstxMemVirtualRangeDesc_t memRangeDesc{deviceId, ptrHeap, 500};                 // 生成一个虚拟内存结构体
    heapDesc.typeSpecificDesc = reinterpret_cast<void const*>(&memRangeDesc);
    auto HeapHandle = MstxMemHeapRegisterFunc(domainHandle, &heapDesc);
    EXPECT_EQ(PTACachingPoolTrace::GetInstance().memUsageMp_[deviceId].totalReserved, 500);
 
    void const* ptr = reinterpret_cast<void const*>(123);
    memRangeDesc = {deviceId, ptr, 50};
    mstxMemRegionsRegisterBatch_t desc;
    desc.regionCount = 1;
    desc.regionDescArray = reinterpret_cast<const void *>(&memRangeDesc);
    auto mstxMemRegionHandle = mstxMemRegionHandle_t {};
    auto handleArrayOut = &mstxMemRegionHandle;
    desc.regionHandleArrayOut = handleArrayOut;
    MstxMemRegionsRegisterFunc(domainHandle, &desc);
    EXPECT_EQ(PTACachingPoolTrace::GetInstance().memUsageMp_[deviceId].totalAllocated, 50);
 
    mstxMemRegionsUnregisterBatch_t unregisterBatch;
    unregisterBatch.refCount = 1;
    auto mstxMemRegionRef = new mstxMemRegionRef_t {};
 
    mstxMemRegionRef->pointer = ptr;
    unregisterBatch.refArray = mstxMemRegionRef;
 
    MstxMemRegionsUnregisterFunc(domainHandle, &unregisterBatch);
    EXPECT_EQ(PTACachingPoolTrace::GetInstance().memUsageMp_[deviceId].totalAllocated, 0);
 
    MstxMemHeapUnregisterFunc(domainHandle, static_cast<mstxMemHeapHandle_t>(const_cast<void *>(ptrHeap)));
    EXPECT_EQ(PTACachingPoolTrace::GetInstance().memUsageMp_[deviceId].totalReserved, 0);
}

TEST(MemPoolTraceTest, PTACachingPoolTraceReleaseRegionHandleMpNull)
{
    mstxDomainRegistration_t domainRegistration{};
    mstxDomainHandle_t  domainHandle = &domainRegistration;
    mstxMemRegionsUnregisterBatch_t desc;
    void const* ptr = reinterpret_cast<void const*>(123);
    mstxMemRegionRef_t memRangeDesc = {mstxMemRegionRefType::MSTX_MEM_REGION_REF_TYPE_HANDLE, ptr};
    desc.refCount = 1;
    desc.refArray = &memRangeDesc;
    PTACachingPoolTrace::GetInstance().regionHandleMp_ = {};
    PTACachingPoolTrace::GetInstance().Release(domainHandle, &desc);
}

TEST(MemPoolTraceTest, PTACachingPoolTraceCreateDomainReturnNull)
{
    EXPECT_EQ(PTACachingPoolTrace::GetInstance().CreateDomain("msmemscope"), nullptr);
}

TEST(MemPoolTraceTest, PTAWorkspacePoolTraceTestPtaHeapRegisterAndRegionRegister)
{
    auto domainHandle = MstxDomainCreateAFunc("ptaWorkspace");
    EXPECT_EQ(domainHandle, PTAWorkspacePoolTrace::GetInstance().ptaWorkspaceDomain_);
 
    mstxMemHeapDesc_t heapDesc;
    uint32_t deviceId = 1;
    void const* ptrHeap = reinterpret_cast<void const*>(123);
    mstxMemVirtualRangeDesc_t memRangeDesc{deviceId, ptrHeap, 500};                 // 生成一个虚拟内存结构体
    heapDesc.typeSpecificDesc = reinterpret_cast<void const*>(&memRangeDesc);
    auto HeapHandle = MstxMemHeapRegisterFunc(domainHandle, &heapDesc);
    EXPECT_EQ(PTAWorkspacePoolTrace::GetInstance().memUsageMp_[deviceId].totalReserved, 500);
 
    void const* ptr = reinterpret_cast<void const*>(123);
    memRangeDesc = {deviceId, ptr, 50};
    mstxMemRegionsRegisterBatch_t desc;
    desc.regionCount = 1;
    desc.regionDescArray = reinterpret_cast<const void *>(&memRangeDesc);
    auto mstxMemRegionHandle = mstxMemRegionHandle_t {};
    auto handleArrayOut = &mstxMemRegionHandle;
    desc.regionHandleArrayOut = handleArrayOut;
    MstxMemRegionsRegisterFunc(domainHandle, &desc);
    EXPECT_EQ(PTAWorkspacePoolTrace::GetInstance().memUsageMp_[deviceId].totalAllocated, 50);
 
    mstxMemRegionsUnregisterBatch_t unregisterBatch;
    unregisterBatch.refCount = 1;
    auto mstxMemRegionRef = new mstxMemRegionRef_t {};
 
    mstxMemRegionRef->pointer = ptr;
    unregisterBatch.refArray = mstxMemRegionRef;
 
    MstxMemRegionsUnregisterFunc(domainHandle, &unregisterBatch);
    EXPECT_EQ(PTAWorkspacePoolTrace::GetInstance().memUsageMp_[deviceId].totalAllocated, 0);
 
    MstxMemHeapUnregisterFunc(domainHandle, static_cast<mstxMemHeapHandle_t>(const_cast<void *>(ptrHeap)));
    EXPECT_EQ(PTAWorkspacePoolTrace::GetInstance().memUsageMp_[deviceId].totalReserved, 0);
}

TEST(MemPoolTraceTest, PTAWorkspacePoolTraceReleaseRegionHandleMpNull)
{
    mstxDomainRegistration_t domainRegistration{};
    mstxDomainHandle_t  domainHandle = &domainRegistration;
    mstxMemRegionsUnregisterBatch_t desc;
    void const* ptr = reinterpret_cast<void const*>(123);
    mstxMemRegionRef_t memRangeDesc = {mstxMemRegionRefType::MSTX_MEM_REGION_REF_TYPE_HANDLE, ptr};
    desc.refCount = 1;
    desc.refArray = &memRangeDesc;
    PTAWorkspacePoolTrace::GetInstance().regionHandleMp_ = {};
    PTAWorkspacePoolTrace::GetInstance().Release(domainHandle, &desc);
}

TEST(MemPoolTraceTest, PTAWorkspacePoolTraceCreateDomainReturnNull)
{
    EXPECT_EQ(PTAWorkspacePoolTrace::GetInstance().CreateDomain("msmemscope"), nullptr);
}