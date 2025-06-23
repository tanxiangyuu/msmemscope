// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "gtest/gtest.h"
#include "secodeFuzz.h"
#include "securec.h"

constexpr int32_t ARG_NUM = 2;
constexpr int32_t FUZZ_TEST_TIMES = 1000000;
int32_t g_fuzzRunTime = 0;

GTEST_API_ int main(int argc, char **argv)
{
    DT_Set_Report_Path("./test/fuzz_report");
    DT_SetEnableFork(1);
    if (argc == ARG_NUM) {
        if (sscanf_s(argv[1], "%d", &g_fuzzRunTime) != 0) {
            printf("sscanf_s failed");
        }
    } else {
        g_fuzzRunTime = FUZZ_TEST_TIMES;
    }
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}