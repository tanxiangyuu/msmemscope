/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */
#include "event_trace/mstx_hooks/mstx_inject.h"

#include <gtest/gtest.h>

using namespace MemScope;

TEST(MstxTest, MstxMarkAFuncTest) {
    const char* msg = "Test Message";
    aclrtStream stream = nullptr;
    MstxMarkAFunc(msg, stream);
}


TEST(MstxTest, MstxRangeStartAFuncTest) {
    const char* msg = "Range Start Message";
    aclrtStream stream = nullptr;

    uint64_t rangeId = MstxRangeStartAFunc(msg, stream);

    EXPECT_GT(rangeId, 0);  // 假设 rangeId 应大于 0
}

TEST(MstxTest, MstxRangeEndFuncTest) {
    uint64_t rangeId = 12345;  // 假设一个有效的 rangeId

    MstxRangeEndFunc(rangeId);
}

extern "C" int __attribute__((visibility("default"))) InitInjectionMstx(MstxGetModuleFuncTableFunc getFuncTable);

// 1. 测试 getFuncTable 为 nullptr 的情况
TEST(InitInjectionMstxTest, NullGetFuncTable) {
    EXPECT_EQ(InitInjectionMstx(nullptr), MSTX_FAIL);
}

// 2. 测试 getFuncTable 返回失败的情况
TEST(InitInjectionMstxTest, GetFuncTableFail) {
    auto mockGetFuncTable = [](mstxFuncModule, MstxFuncTable*, unsigned int*) -> int {
        return MSTX_FAIL; // 模拟返回失败
    };
    EXPECT_EQ(InitInjectionMstx(mockGetFuncTable), MSTX_FAIL);
}

// 3. 测试 getFuncTable 设置 outTable 为空的情况
TEST(InitInjectionMstxTest, OutTableIsNull) {
    auto mockGetFuncTable = [](mstxFuncModule, MstxFuncTable* outTable, unsigned int* outSize) -> int {
        *outTable = nullptr;      // 模拟 outTable 为空
        *outSize = static_cast<unsigned int>(mstxImplCoreFuncId::MSTX_API_CORE_RANGE_END);
        return MSTX_SUCCESS;
    };
    EXPECT_EQ(InitInjectionMstx(mockGetFuncTable), MSTX_FAIL);
}

// 4. 测试成功初始化的情况
TEST(InitInjectionMstxTest, SuccessfulInitialization) {
    auto mockGetFuncTable = [](mstxFuncModule, MstxFuncTable* outTable, unsigned int* outSize) -> int {
        static MstxFuncPointer funcArray[4];  // 模拟一个大小为 3 的函数表
        static MstxFuncPointer* funcTable[] = {funcArray, funcArray + 1, funcArray + 2, funcArray + 3};
        *outTable = funcTable;
        *outSize = static_cast<unsigned int>(mstxImplCoreFuncId::MSTX_API_CORE_RANGE_END);
        return MSTX_SUCCESS;
    };

    EXPECT_EQ(InitInjectionMstx(mockGetFuncTable), MSTX_SUCCESS);
}

TEST(MstxTest, ReportDomainTest) {
    MstxDomainCreateAFunc("test");
}

TEST(MstxTest, ReportHeapRegisterTest) {
    mstxDomainHandle_t msmemscope = MstxDomainCreateAFunc("msmemscope");
    mstxMemHeapDesc_t heapDesc;
    void const* ptr = reinterpret_cast<void const*>(123);
    mstxMemVirtualRangeDesc_t memRangeDesc{1, ptr, 1};
    heapDesc.typeSpecificDesc = reinterpret_cast<void const*>(&memRangeDesc);
    MstxMemHeapRegisterFunc(msmemscope, &heapDesc);
}

TEST(MstxTest, ReportHeapUnregisterTest) {
    mstxDomainHandle_t msmemscope = MstxDomainCreateAFunc("msmemscope");
    mstxMemHeapHandle_t heap;
    MstxMemHeapUnregisterFunc(msmemscope, heap);
    mstxMemHeapDesc_t heapDesc;
    void const* ptr = reinterpret_cast<void const*>(123);
    mstxMemVirtualRangeDesc_t memRangeDesc{1, ptr, 1};
    heapDesc.typeSpecificDesc = reinterpret_cast<void const*>(&memRangeDesc);
    MstxMemHeapRegisterFunc(msmemscope, &heapDesc);
    MstxMemHeapUnregisterFunc(msmemscope, reinterpret_cast<mstxMemHeapHandle_t>(const_cast<void*>(ptr)));
}

TEST(MstxTest, ReportRegionsHandleTest) {
    mstxDomainHandle_t msmemscope = MstxDomainCreateAFunc("msmemscope");
    void const* ptr = reinterpret_cast<void const*>(123);
    mstxMemVirtualRangeDesc_t memRangeDesc{1, ptr, 1};
    mstxMemRegionsRegisterBatch_t desc;
    desc.regionCount = 1;
    desc.regionDescArray = reinterpret_cast<const void *>(&memRangeDesc);
    MstxMemRegionsRegisterFunc(msmemscope, &desc);


    mstxMemRegionsUnregisterBatch_t unregisterBatch;
    unregisterBatch.refCount = 1;
    mstxMemRegionRef_t regionRef[1] = {};
    regionRef[0].refType = MSTX_MEM_REGION_REF_TYPE_POINTER;
    regionRef[0].pointer = ptr;
    unregisterBatch.refArray = regionRef;

    MstxMemRegionsUnregisterFunc(msmemscope, &unregisterBatch);
}