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

#include "event_trace/ascend_hal.h"

#include <gtest/gtest.h>

TEST(AscendHalStubTest, HalMemAllocInnerReturnsSuccess) {
    void* ptr = nullptr;
    unsigned long long size = 1024;
    unsigned long long flag = 0;

    drvError_t result = halMemAllocInner(&ptr, size, flag);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST(AscendHalStubTest, HalMemFreeInnerReturnsSuccess) {
    void* ptr = nullptr;

    drvError_t result = halMemFreeInner(ptr);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}