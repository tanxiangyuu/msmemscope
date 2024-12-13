// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#ifndef MSLEAKS_MSTX_MANAGER_H
#define MSLEAKS_MSTX_MANAGER_H

#include <atomic>
#include <cstdint>

namespace Leaks {
/*
* mstx_inject仅仅做接口的转发调用，实际的上报信息组装由Manager来完成，避免接口变动时改动过大
*/
class MstxManager {
public:
    static MstxManager& GetInstance()
    {
        static MstxManager instance;
        return instance;
    }
    MstxManager(const MstxManager&) = delete;
    MstxManager& operator=(const MstxManager&) = delete;

    void ReportMarkA(const char* msg, int32_t stream);
    uint64_t ReportRangeStart(const char* msg, int32_t stream);
    void ReportRangeEnd(uint64_t id);

private:
    MstxManager()
    {
        rangeId_ = 1;
    }
    ~MstxManager() = default;
    uint64_t GetRangeId();

private:
    std::atomic<uint64_t> rangeId_;
    const int onlyMarkId_ = 0;
};

}

#endif