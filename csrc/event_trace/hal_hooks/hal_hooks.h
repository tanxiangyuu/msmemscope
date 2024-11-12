// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef LEAKS_HOOKS_HAL_HOOKS_H
#define LEAKS_HOOKS_HAL_HOOKS_H

#include "ascend_hal.h"

constexpr uint64_t MEM_VIRT_BIT = 10;
constexpr uint64_t MEM_VIRT_WIDTH = 4;
constexpr uint64_t MEM_DEV_VAL = 0x1;
constexpr uint64_t MEM_HOST_VAL = 0x2;
constexpr uint64_t MEM_HOST = MEM_HOST_VAL << MEM_VIRT_BIT;
constexpr uint64_t MEM_DEV = MEM_DEV_VAL << MEM_VIRT_BIT;
constexpr uint64_t MEM_VIRT_MASK = ((1U << MEM_VIRT_WIDTH) - 1) << MEM_VIRT_BIT;

inline int32_t GetMallocModuleId(unsigned long long flag)
{
    return flag & MEM_VIRT_MASK;
}

/// Interfaces to be hooked use C ABI
extern "C" {
drvError_t halMemAlloc(void **pp, unsigned long long size, unsigned long long flag);
drvError_t halMemFree(void *pp);
}  // extern "C"

#endif