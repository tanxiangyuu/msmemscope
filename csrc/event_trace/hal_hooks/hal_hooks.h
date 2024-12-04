// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef LEAKS_HOOKS_HAL_HOOKS_H
#define LEAKS_HOOKS_HAL_HOOKS_H

#include "ascend_hal.h"
#include "event_report.h"

constexpr uint64_t MEM_VIRT_BIT = 10;
constexpr uint64_t MEM_SVM_VAL = 0x0;
constexpr uint64_t MEM_DEV_VAL = 0x1;
constexpr uint64_t MEM_HOST_VAL = 0x2;
constexpr uint64_t MEM_DVPP_VAL = 0x3;

/// Interfaces to be hooked use C ABI
extern "C" {
drvError_t halMemAlloc(void **pp, unsigned long long size, unsigned long long flag);
drvError_t halMemFree(void *pp);
Leaks::MemOpSpace GetMemOpSpace(unsigned long long flag);
}  // extern "C"

#endif