// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "describe_trace.h"
#include <cstring>
#include <algorithm>
#include "securec.h"
#include "call_stack.h"
#include "event_report.h"
#include "record_info.h"
#include "log.h"


namespace Leaks {

std::string DescribeTrace::GetDescribe()
{
    std::string res;
    auto tid = Utility::GetTid();
    for (auto s : describe_[tid]) {
        res += "@" + s;
    }
    return res;
}

bool DescribeTrace::IsRepeat(uint64_t threadId, std::string owner)
{
    for (auto s : describe_[threadId]) {
        if (s == owner) {
            return true;
        }
    }
    return false;
}

void DescribeTrace::DescribeAddr(uint64_t addr, std::string owner)
{
    auto tid = Utility::GetTid();
    owner = "@" + owner;
    AddrInfo info;
    info.addr = addr;
    if (strncpy_s(info.owner, sizeof(info.owner), owner.c_str(), sizeof(info.owner) - 1) != EOK) {
        CLIENT_ERROR_LOG("strncpy_s FAILED");
    }
    info.owner[sizeof(info.owner) - 1] = '\0';
    EventReport::Instance(CommType::SOCKET).ReportAddrInfo(info);
    return;
}

void DescribeTrace::AddDescribe(std::string owner)
{
    auto tid = Utility::GetTid();
    if (IsRepeat(tid, owner)) {
        CLIENT_ERROR_LOG("Cannot add duplicate tags " + owner);
        return;
    }
    if (describe_[tid].size() >= maxSize) {
        CLIENT_ERROR_LOG("The current thread label exceeds " + std::to_string(maxSize));
        return;
    }
    describe_[tid].emplace_back(owner);
}

void DescribeTrace::EraseDescribe(std::string owner)
{
    auto tid = Utility::GetTid();
    size_t i;
    size_t siz = describe_[tid].size();
    for (i = 0; i < siz; i++) {
        if (owner == describe_[tid][i]) {
            describe_[tid].erase(describe_[tid].begin() + i);
            break;
        }
    }
    if (i == siz) {
        CLIENT_ERROR_LOG("Tag " + owner + " not found");
    }
}

}