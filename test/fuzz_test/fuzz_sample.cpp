/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */

#include "fuzz.h"
#include "securec.h"

constexpr uint8_t ERROR_NUM = 11;
constexpr uint32_t LOOP_NUM = 1000000;
void Func1(int num)
{
    if (num == 0x2347) {
        int* p = nullptr;
        *p = ERROR_NUM;
    }
}

void Func2(int num)
{
    if (num == 0x2347) {
        int a[10] = {0};
        a[ERROR_NUM] = a[0];
    }
}

void Func3(int num)
{
    if (num == 0x2347) {
        for (int i = 0; i < LOOP_NUM; i++) {
            for (int j = 0; j < LOOP_NUM; j++) {
                int a = i * j;
            }
        }
    }
}

TEST(fuzz_test_sample1, fuzz_test_sample1)
{
    char testApi[] = "fuzz_test_sample1";
    DT_FUZZ_START(0, g_fuzzRunTime, testApi, 0)
    {
        printf("\rFuzzing %d times", fuzzSeed + fuzzi);
        s32 number = *(s32 *)DT_SetGetS32(&g_Element[0], 0x2345);
        Func1(number);
    }
    DT_FUZZ_END()
}

TEST(fuzz_test_sample2, fuzz_test_sample2)
{
    char testApi[] = "fuzz_test_sample2";
    DT_FUZZ_START(0, g_fuzzRunTime, testApi, 0)
    {
        printf("\rFuzzing %d times", fuzzSeed + fuzzi);
        s32 number = *(s32 *)DT_SetGetS32(&g_Element[0], 0x2345);
        Func2(number);
    }
    DT_FUZZ_END()
}

TEST(fuzz_test_sample3, fuzz_test_sample3)
{
    DT_Set_TimeOut_Second(1);
    char testApi[] = "fuzz_test_sample3";
    DT_FUZZ_START(0, g_fuzzRunTime, testApi, 0)
    {
        printf("\rFuzzing %d times", fuzzSeed + fuzzi);
        s32 number = *(s32 *)DT_SetGetS32(&g_Element[0], 0x2345);
        Func3(number);
    }
    DT_FUZZ_END()
}