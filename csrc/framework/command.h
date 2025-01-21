// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef FRAMEWORK_COMMAND_H
#define FRAMEWORK_COMMAND_H

#include <vector>
#include <string>
#include <memory>
#include "analysis/analyzer_factory.h"
#include "analysis/mstx_analyzer.h"
#include "analysis/dump_record.h"
#include "analysis/trace_record.h"
#include "analysis/stepinter_analyzer.h"

namespace Leaks {

constexpr double MICROSEC = 1000000.0;

void RecordHandler(const ClientId &clientId, const EventRecord &record, AnalyzerFactory &analyzerfactory);
void DumpHandler(const ClientId &clientId, DumpRecord &dump, const EventRecord &record);

// Command类主要针对解析后的命令进行处理，是串接流程的主要类
class Command {
public:
    explicit Command(const AnalysisConfig &config) : config_{config} {}
    void Exec(const std::vector<std::string> &execParams) const;
    void StepInterCompare(const std::vector<std::string> &paths) const;
private:
    AnalysisConfig config_;
};

}

#endif
