// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 
#ifndef DATA_BASE_H
#define DATA_BASE_H

#include <cstdint>

namespace Leaks {

enum class DataType : uint8_t {
    LEAKS_EVENT = 0,
    PYTHON_TRACE_EVENT = 1,
};

class DataBase {
public:
    virtual ~DataBase() = default;
    explicit DataBase(DataType type) : dataType(type) {}
    DataType dataType;
};

}

#endif