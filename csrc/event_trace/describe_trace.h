/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */
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