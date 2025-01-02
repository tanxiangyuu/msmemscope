// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "analyzer_factory.h"
#include "hal_analyzer.h"
#include "mstx_analyzer.h"
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
            MstxAnalyzer::Instance().RegisterAnalyzer(analyzers[type]);
        } else {
            return nullptr;
        }
    }
    return analyzers[type];
}

}