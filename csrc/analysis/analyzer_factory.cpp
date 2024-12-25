// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "analyzer_factory.h"
#include "hal_analyzer.h"
#include "stepinner_analyzer.h"

namespace Leaks {

AnalyzerFactory::AnalyzerFactory(const AnalysisConfig &config)
{
    config_ = config;
}

std::shared_ptr<AnalyzerBase> AnalyzerFactory::CreateAnalyzer(const RecordType &type)
{
    if (analyzers.find(type) == analyzers.end()) {
        if (type == RecordType::MEMORY_RECORD) {
            analyzers[type] = std::make_shared<HalAnalyzer>(config_);
        } else if (type == RecordType::TORCH_NPU_RECORD) {
            analyzers[type] = std::make_shared<StepInnerAnalyzer>(config_);
            npuRegisterStatus = RegisterStatus::RECORD_BUT_NO_REGEISTER;
        } else {
            return nullptr;
        }
    }
    return analyzers[type];
}

std::shared_ptr<std::list<std::shared_ptr<AnalyzerBase>>> AnalyzerFactory::ReturnRegisterList()
{
    auto registerListptr = std::make_shared<std::list<std::shared_ptr<AnalyzerBase>>>();
    // 返回仍未注册的analyzer类，在这里补充新的分析类
    if (npuRegisterStatus == RegisterStatus::RECORD_BUT_NO_REGEISTER) {
        npuRegisterStatus = RegisterStatus::REGISTER_ALREADY;
        (*registerListptr).push_back(analyzers[RecordType::TORCH_NPU_RECORD]);
    }

    if (registerListptr->empty()) {
        return nullptr;
    } else {
        return registerListptr;
    }
}

}