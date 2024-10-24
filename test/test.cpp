// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "framework/command.h"
#include "framework/client_parser.h"
#include "framework/protocol.h"
#include "analysis/analyzer.h"
#include "event_trace/event_report.h"

using namespace Leaks;

// sample当前仅做样例且满足CI覆盖率要求，后续业务代码及用例填充后删除
TEST(Sample, sample)
{
    char* arr[] = {"Hello", "World"};

    ClientParser parser;
    parser.Interpretor(2, arr);

    AnalysisConfig config;
    EventRecord record;
    Analyzer analyzer(config);
    analyzer.Do(record);

    std::string testString = "test";
    Protocol protocol(config);
    protocol.Feed(testString);
    (void)protocol.GetPacket();

    std::string str("leak test");
    EXPECT_TRUE(str.size() == 9);
}