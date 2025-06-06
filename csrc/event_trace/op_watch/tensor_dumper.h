// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef TENSOR_DUMPER_H
#define TENSOR_DUMPER_H

#include <string>
#include <mutex>
#include <unordered_map>
#include "tensor_monitor.h"
#include "record_info.h"

namespace Leaks {

class TensorDumper {
public:
    TensorDumper(const TensorDumper&) = delete;
    TensorDumper& operator=(const TensorDumper&) = delete;

    static TensorDumper& GetInstance()
    {
        static TensorDumper instance;
        return instance;
    }

    bool DumpOneTensor(const MonitoredTensor& tensor, std::string& fileName);
    void Dump(const std::string &op, OpEventType eventType);
    void SetDumpNums(uint64_t ptr, int32_t dumpNums);
    int32_t GetDumpNums(uint64_t ptr);
    void DeleteDumpNums(uint64_t ptr);
    void SetDumpName(uint64_t ptr, std::string name);
    std::string GetDumpName(uint64_t ptr);
    void DeleteDumpName(uint64_t ptr);

private:
    TensorDumper();
    ~TensorDumper();

    bool IsDumpFullContent();

    bool DumpTensorBinary(const std::vector<char> &hostData, std::string& fileName);
    bool DumpTensorHashValue(const std::vector<char> &hostData, std::string& fileName);

private:
    bool fullContent_;
    std::string dumpDir_;
    FILE *csvFile_ = nullptr; // 仅落盘哈希值时的csv文件指针
    std::mutex mutex_;
    std::mutex mapMutex_;
    std::unordered_map<uint64_t, std::string> dumpNameMap_;
    std::unordered_map<uint64_t, int32_t> dumpNumsMap_;
    std::string fileName_;
};

void CleanFileName(std::string& fileName);
}
#endif
