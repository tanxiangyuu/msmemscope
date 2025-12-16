/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <string>
#include <memory>
#include "mstx_analyzer.h"
#include "record_info.h"
#include "config_info.h"

using namespace MemScope;

TEST(MstxAnalyzerTest, do_mstx_record_expect_success) {
    ClientId clientId = 0;
    auto mstxRecordStart = MstxRecord {};
    mstxRecordStart.markType = MarkType::RANGE_START_A;
    mstxRecordStart.stepId = 1;
    mstxRecordStart.streamId = 123;

    auto mstxRecordEnd = MstxRecord {};
    mstxRecordEnd.markType = MarkType::RANGE_END;
    mstxRecordEnd.stepId = 1;
    mstxRecordEnd.streamId = 123;

    auto mstxRecordMark = MstxRecord {};
    mstxRecordMark.markType = MarkType::MARK_A;
    mstxRecordMark.stepId = 0;
    mstxRecordMark.streamId = 123;

    EXPECT_TRUE(MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordStart));
    EXPECT_TRUE(MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordEnd));
    EXPECT_TRUE(MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordMark));
}

TEST(MstxAnalyzerTest, do_analyzer_register_and_unregister_expect_success) {
    MstxAnalyzer::Instance().Subscribe(MstxEventSubscriber::STEP_INNER_ANALYZER, nullptr);
    MstxAnalyzer::Instance().UnSubscribe(MstxEventSubscriber::STEP_INNER_ANALYZER);
}

TEST(MstxAnalyzerTest, do_analyzer_notify_expect_success) {
    MstxAnalyzer::Instance().Subscribe(MstxEventSubscriber::STEP_INNER_ANALYZER, nullptr);

    ClientId clientId = 0;
    auto mstxRecordStart = MstxRecord {};
    mstxRecordStart.markType = MarkType::RANGE_START_A;
    mstxRecordStart.stepId = 1;
    mstxRecordStart.streamId = 123;

    EXPECT_TRUE(MstxAnalyzer::Instance().RecordMstx(clientId, mstxRecordStart));
    MstxAnalyzer::Instance().UnSubscribe(MstxEventSubscriber::STEP_INNER_ANALYZER);
}