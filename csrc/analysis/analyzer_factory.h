// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef ANALYZER_FACTORY_H
#define ANALYZER_FACTORY_H

#include <list>
#include <memory>
#include "analysis/analyzer_base.h"

namespace Leaks {
/*
 * AnalyzerFactory类主要功能：
 * 1. 维护并提供各种analyzer分析器的指针
   2. 为mstx注册观察者提供对应指针
*/

// 注册状态
enum class RegisterStatus : uint8_t {
    HAVE_NOT_RECORD = 0U,
    RECORD_BUT_NO_REGEISTER,
    REGISTER_ALREADY
};


class AnalyzerFactory {
public:
    explicit AnalyzerFactory(const AnalysisConfig &config);
    std::shared_ptr<AnalyzerBase> CreateAnalyzer(const RecordType &type);
    std::shared_ptr<std::list<std::shared_ptr<AnalyzerBase>>> ReturnRegisterList();
private:
    std::unordered_map<RecordType, std::shared_ptr<AnalyzerBase>> analyzers;
    RegisterStatus npuRegisterStatus;  // 注册状态标识，如果添加分析器需添加对应的类型
    AnalysisConfig config_;
};

}

#endif