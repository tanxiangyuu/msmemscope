// Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

#ifndef COMM_DEF_H
#define COMM_DEF_H

namespace Leaks {

using ClientId = std::size_t;

enum class LeaksCommType {
    SHARED_MEMORY,
    DOMAIN_SOCKET,
    MEMORY_DEBUG
};

constexpr size_t SHM_SIZE = 200 * 1024 * 1024;
constexpr size_t SHM_S2C_SIZE = 4 * 1024 * 1024;

}

#endif