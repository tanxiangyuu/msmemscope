// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "event_trace/mstx_hooks/mstx_inject.h"

#include <gtest/gtest.h>

using namespace Leaks;

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