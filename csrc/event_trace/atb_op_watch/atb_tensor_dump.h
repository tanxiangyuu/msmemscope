// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef ATB_TENSOR_DUMP_H
#define ATB_TENSOR_DUMP_H

#include <string>

namespace Leaks {

// 对于dump功能，只需要tensor的device地址和大小即可
struct Tensor {
    void* data;
    uint64_t dataSize;
};

class ATBTensorDump {
public:
    ATBTensorDump(const ATBTensorDump&) = delete;
    ATBTensorDump& operator=(const ATBTensorDump&) = delete;

    static ATBTensorDump& GetInstance()
    {
        static ATBTensorDump instance;
        return instance;
    }

    bool Dump(const Tensor& tensor, std::string& fileName);

private:
    ATBTensorDump() = default;
    ~ATBTensorDump() = default;
};
}
#endif