// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef LEAKS_UTILITY_UMASK_GUARD_H
#define LEAKS_UTILITY_UMASK_GUARD_H

#include <sys/stat.h>
#include <sys/types.h>

namespace Utility {

class UmaskGuard {
public:
    explicit UmaskGuard(mode_t mask) noexcept : oldUmask_(umask(mask)) {}
    UmaskGuard(const UmaskGuard &) = delete;
    UmaskGuard &operator=(const UmaskGuard &) = delete;
    UmaskGuard(UmaskGuard &&) = delete;
    UmaskGuard &operator=(UmaskGuard &&) = delete;
    ~UmaskGuard() { umask(oldUmask_); }

private:
    mode_t oldUmask_;
};

}  // namespace Utility

#endif
