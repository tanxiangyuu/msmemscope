// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "describe_trace.h"
#include <cstring>
#include <algorithm>
#include "securec.h"
#include "call_stack.h"
#include "event_report.h"
#include "record_info.h"
#include "log.h"
#include "ustring.h"

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
    Utility::ToSafeString(owner);
    for (auto s : describe_[threadId]) {
        if (s == owner) {
            return true;
        }
    }
    return false;
}

void DescribeTrace::DescribeAddr(uint64_t addr, std::string owner)
{
    Utility::ToSafeString(owner);
    RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<AddrInfo>(
        TLVBlockType::ADDR_OWNER, "@" + owner);
    AddrInfo* info = buffer.Cast<AddrInfo>();
    info->subtype = RecordSubType::USER_DEFINED;
    info->addr = addr;
    EventReport::Instance(CommType::SOCKET).ReportAddrInfo(buffer);
    return;
}

void DescribeTrace::AddDescribe(std::string owner)
{
    auto tid = Utility::GetTid();
    Utility::ToSafeString(owner);
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
    Utility::ToSafeString(owner);
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