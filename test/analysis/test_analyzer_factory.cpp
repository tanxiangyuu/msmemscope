// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <string>
#include "analyzer_factory.h"
#include "record_info.h"
#include "config_info.h"

using namespace Leaks;

TEST(AnalyzerFactoryTest, AnalyzerFactoryConstruct) {
    AnalysisConfig analysisConfig;
    AnalyzerFactory analyzerfactory{analysisConfig};
}

TEST(AnalyzerFactoryTest, Do_createHalAnalyzer_except_success) {
    AnalysisConfig analysisConfig;
    AnalyzerFactory analyzerfactory{analysisConfig};

    RecordType type = RecordType::MEMORY_RECORD;
    EXPECT_NE(analyzerfactory.CreateAnalyzer(type), nullptr);
}

TEST(AnalyzerFactoryTest, Do_createNpuAnalyzer_except_success) {
    AnalysisConfig analysisConfig;
    AnalyzerFactory analyzerfactory{analysisConfig};

    RecordType type = RecordType::TORCH_NPU_RECORD;
    EXPECT_NE(analyzerfactory.CreateAnalyzer(type), nullptr);
}

TEST(AnalyzerFactoryTest, Do_createKernelAnalyzer_except_success) {
    AnalysisConfig analysisConfig;
    AnalyzerFactory analyzerfactory{analysisConfig};

    RecordType type = RecordType::KERNEL_LAUNCH_RECORD;
    EXPECT_EQ(analyzerfactory.CreateAnalyzer(type), nullptr);
}

TEST(AnalyzerFactoryTest, Do_createAclAnalyzer_except_success) {
    AnalysisConfig analysisConfig;
    AnalyzerFactory analyzerfactory{analysisConfig};

    RecordType type = RecordType::ACL_ITF_RECORD;
    EXPECT_EQ(analyzerfactory.CreateAnalyzer(type), nullptr);
}

TEST(AnalyzerFactoryTest, Do_createAnalyzer_get_unsupportedType_except_failed) {
    AnalysisConfig analysisConfig;
    AnalyzerFactory analyzerfactory{analysisConfig};

    RecordType type = RecordType::ACL_ITF_RECORD;
    analyzerfactory.CreateAnalyzer(type);
    EXPECT_EQ(analyzerfactory.CreateAnalyzer(type), nullptr);
}
