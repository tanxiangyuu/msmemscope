// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef LEAKS_HOOKS_HAL_HOOKS_H
#define LEAKS_HOOKS_HAL_HOOKS_H

#include "ascend_hal.h"
#include "event_report.h"

/// Interfaces to be hooked use C ABI
extern "C" {
drvError_t halMemAlloc(void **pp, unsigned long long size, unsigned long long flag);
drvError_t halMemFree(void *pp);
}  // extern "C"

#endif