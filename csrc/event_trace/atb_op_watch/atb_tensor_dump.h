// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef ATB_TENSOR_DUMP_H
#define ATB_TENSOR_DUMP_H

#include <string>
#include <mutex>

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
    ATBTensorDump();
    ~ATBTensorDump();

    bool IsDumpFullContent();

    bool DumpTensorBinary(const std::vector<char> &hostData, std::string& fileName);
    bool DumpTensorMD5(const std::vector<char> &hostData, std::string& fileName);

private:
    bool fullContent_;
    std::string dumpDir_;
    FILE *csvFile_ = nullptr; // 仅落盘哈希值时的csv文件指针
    std::mutex mutex_;
};

void CleanFileName(std::string& fileName);
}
#endif