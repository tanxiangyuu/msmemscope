// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <string>
#include <memory>
#include "analyzer_factory.h"
#include "mstx_analyzer.h"
#include "record_info.h"
#include "config_info.h"

using namespace Leaks;

TEST(MstxAnalyzerTest, do_mstx_record_expect_success) {
    ClientId clientId = 0;
    auto mstxRecordStart = MstxRecord {};
    mstxRecordStart.markType = MarkType::RANGE_START_A;
    mstxRecordStart.rangeId = 1;
    mstxRecordStart.streamId = 123;

    auto mstxRecordEnd = MstxRecord {};
    mstxRecordEnd.markType = MarkType::RANGE_END;
    mstxRecordEnd.rangeId = 1;
    mstxRecordEnd.streamId = 123;

    auto mstxRecordMark = MstxRecord {};
    mstxRecordMark.markType = MarkType::MARK_A;
    mstxRecordMark.rangeId = 0;
    mstxRecordMark.streamId = 123;

    EXPECT_TRUE(MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordStart));
    EXPECT_TRUE(MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordEnd));
    EXPECT_TRUE(MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordMark));
}

TEST(MstxAnalyzerTest, do_analyzer_register_and_unregister_expect_success) {
    AnalysisConfig analysisConfig;
    AnalyzerFactory analyzerfactory{analysisConfig};
    RecordType type = RecordType::TORCH_NPU_RECORD;
    std::shared_ptr<AnalyzerBase> analyzer = analyzerfactory.CreateAnalyzer(type);

    MstxAnalyzer::Instance().RegisterAnalyzer(analyzer);
    MstxAnalyzer::Instance().UnregisterAnalyzer(analyzer);
}

TEST(MstxAnalyzerTest, do_analyzer_notify_expect_success) {
    AnalysisConfig analysisConfig;
    AnalyzerFactory analyzerfactory{analysisConfig};
    RecordType type = RecordType::TORCH_NPU_RECORD;
    std::shared_ptr<AnalyzerBase> analyzer = analyzerfactory.CreateAnalyzer(type);

    MstxAnalyzer::Instance().RegisterAnalyzer(analyzer);

    ClientId clientId = 0;
    auto mstxRecordStart = MstxRecord {};
    mstxRecordStart.markType = MarkType::RANGE_START_A;
    mstxRecordStart.rangeId = 1;
    mstxRecordStart.streamId = 123;

    EXPECT_TRUE(MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordStart));
}