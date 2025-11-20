// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#ifndef DESCRIBE_TRACE_H
#define DESCRIBE_TRACE_H

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <iostream>
#include <vector>

namespace MemScope {

class DescribeTrace {
public:
    static DescribeTrace& GetInstance()
    {
        static DescribeTrace instance;
        return instance;
    }
    DescribeTrace(const DescribeTrace&) = delete;
    DescribeTrace& operator=(const DescribeTrace&) = delete;

    std::string GetDescribe();
    void AddDescribe(std::string owner);
    void EraseDescribe(std::string owner);
    void DescribeAddr(uint64_t addr, std::string owner);
private:
    DescribeTrace() = default;
    ~DescribeTrace() = default;
    bool IsRepeat(uint64_t threadId, std::string owner);
    std::unordered_map<uint64_t, std::vector<std::string>> describe_;
    const uint8_t maxSize{3};
};

}

#endif