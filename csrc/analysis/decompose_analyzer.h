// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef DECOMPOSE_ANALYZER_H
#define DECOMPOSE_ANALYZER_H

#include "analyzer_base.h"

namespace Leaks {

class DecomposeAnalyzer : public AnalyzerBase {
public:
    static DecomposeAnalyzer& GetInstance();
    void EventHandle(std::shared_ptr<EventBase>& event, MemoryState* state) override;

private:
    explicit DecomposeAnalyzer();
    ~DecomposeAnalyzer() override = default;
    DecomposeAnalyzer(const DecomposeAnalyzer&) = delete;
    DecomposeAnalyzer& operator=(const DecomposeAnalyzer&) = delete;
    DecomposeAnalyzer(DecomposeAnalyzer&& other) = delete;
    DecomposeAnalyzer& operator=(DecomposeAnalyzer&& other) = delete;

    void InitOwner(std::shared_ptr<MemoryEvent>& event, MemoryState* state);
    void UpdateOwnerByAtenAccess(std::shared_ptr<MemoryEvent>& event, MemoryState* state);
    void UpdateOwner(std::shared_ptr<MemoryOwnerEvent>& event, MemoryState* state);

    static const std::string cannStr;
    static const std::string ptaStr;
    static const std::string atbStr;
    static const std::string mindsporeStr;
    static const size_t ptaStrLen;
    static const std::string atenStr;
};

}

#endif