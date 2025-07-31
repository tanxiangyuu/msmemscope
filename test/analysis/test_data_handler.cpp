// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <string>
#include <memory>
#include <cstdio>
#include "config_info.h"
#include "record_info.h"
#include "utils.h"
#include "file.h"
#include "securec.h"
#include "utility/log.h"
#include "data_handler.h"
#include "utility/sqlite_loader.h"

using namespace Leaks;
extern bool g_isDlsymNullptr;

TEST(DataHandler, CsvHandler_Write_LeakRecord)
{
    Config config;
    config.dataFormat = static_cast<uint8_t>(DataFormat::CSV);
    config.enableCStack = false;
    config.enablePyStack = false;
    strncpy_s(config.outputDir, sizeof(config.outputDir) - 1, "./testmsleaks", sizeof(config.outputDir) - 1);

    CsvHandler handler(config, DataType::LEAKS_EVENT);
    handler.Init();
    std::shared_ptr<EventBase> data1 = std::make_shared<EventBase>();
    data1->id = 1;
    data1->eventType = EventBaseType::MALLOC;
    data1->eventSubType = EventSubType::PTA_CACHING;
    data1->name = "obj1";
    data1->timestamp = 123456789;
    data1->pid = 1234;
    data1->tid = 5678;
    data1->device = "1";
    data1->addr = 0x1234;
    data1->attr = "size=1024";
    ASSERT_TRUE(handler.Write(data1));

    config.enablePyStack = true;
    CsvHandler handlerPy(config, DataType::LEAKS_EVENT);
    handlerPy.Init();
    CallStackString stack_;
    data1->pyCallStack = "call_stack_py";
    ASSERT_TRUE(handlerPy.Write(data1));

    CsvHandler handler_(config, DataType::PYTHON_TRACE_EVENT);
    handler_.Init();
    std::shared_ptr<TraceEvent> data2 = std::make_shared<TraceEvent>();
    data2->startTs = 1000;
    data2->endTs = 2000;
    data2->tid = 123;
    data2->pid = 456;
    data2->info = "function_call";
    data2->hash = "hash123";
    ASSERT_TRUE(handler_.Write(data2));
}

TEST(DataHandler, Sqlite3_open)
{
    g_isDlsymNullptr = false;
    sqlite3* db = nullptr;
    std::string path = "./testmsleaks/test.db";
    int rc = Sqlite3Open(path.c_str(), &db);
    EXPECT_EQ(rc, 0);
    Sqlite3Errmsg(db);
}

TEST(DataHandler, DbHandler_Write_LeakRecord)
{
    g_isDlsymNullptr = false;
    Config config;
    config.dataFormat = static_cast<uint8_t>(DataFormat::DB);
    config.enableCStack = true;
    config.enablePyStack = true;
    strncpy_s(config.outputDir, sizeof(config.outputDir) - 1, "./testmsleaks", sizeof(config.outputDir) - 1);
    Utility::CreateDbPath(config, DB_DUMP_FILE);

    std::unique_ptr<DataHandler> handler = MakeDataHandler(config, DataType::LEAKS_EVENT);
    handler->Init();
    std::shared_ptr<EventBase> data1 = std::make_shared<EventBase>();
    data1->id = 1;
    data1->eventType = EventBaseType::MALLOC;
    data1->eventSubType = EventSubType::PTA_CACHING;
    data1->name = "obj1";
    data1->timestamp = 123456789;
    data1->pid = 1234;
    data1->tid = 5678;
    data1->device = "1";
    data1->addr = 0x1234;
    data1->attr = "size=1024";
    data1->cCallStack = "call_stack_c";
    data1->pyCallStack = "call_stack_py";
    ASSERT_TRUE(handler->Write(data1));

    config.enableCStack = false;
    config.enablePyStack = false;
    DbHandler handler_(config, DataType::PYTHON_TRACE_EVENT);
    handler_.Init();
    std::shared_ptr<TraceEvent> data2 = std::make_shared<TraceEvent>();
    data2->startTs = 1000;
    data2->endTs = 2000;
    data2->tid = 123;
    data2->pid = 456;
    data2->info = "function_call";
    data2->hash = "hash123";
    ASSERT_TRUE(handler_.Write(data2));
}

TEST(DataHandler, CsvHandler_InitSetParm_Default)
{
    Config config;
    config.dataFormat = static_cast<uint8_t>(DataFormat::CSV);
    strncpy_s(config.outputDir, sizeof(config.outputDir) - 1, "./testmsleaks", sizeof(config.outputDir) - 1);
    CsvHandler handler(config, static_cast<DataType>(999));
    EXPECT_TRUE(true);
}

TEST(DataHandler, CsvHandler_Write_NullData)
{
    Config config;
    config.dataFormat = static_cast<uint8_t>(DataFormat::CSV);
    strncpy_s(config.outputDir, sizeof(config.outputDir) - 1, "./testmsleaks", sizeof(config.outputDir) - 1);
    CsvHandler handler(config, DataType::LEAKS_EVENT);
    handler.Init();
    ASSERT_FALSE(handler.Write(nullptr));
}

TEST(DataHandler, DbHandler_InitSetParm_Default)
{
    Config config;
    config.dataFormat = static_cast<uint8_t>(DataFormat::DB);
    strncpy_s(config.outputDir, sizeof(config.outputDir) - 1, "./testmsleaks", sizeof(config.outputDir) - 1);
    DbHandler handler(config, static_cast<DataType>(999));
}

TEST(DataHandler, DbHandler_Write_NullData)
{
    g_isDlsymNullptr = false;
    Config config;
    config.dataFormat = static_cast<uint8_t>(DataFormat::DB);
    config.enableCStack = false;
    config.enablePyStack = false;
    strncpy_s(config.outputDir, sizeof(config.outputDir) - 1, "./testmsleaks", sizeof(config.outputDir) - 1);
    Utility::CreateDbPath(config, DB_DUMP_FILE);
    DbHandler handler(config, DataType::LEAKS_EVENT);
    handler.Init();
    ASSERT_FALSE(handler.Write(nullptr));
}

TEST(DataHandler, MakeDataHandler_FALSE)
{
    Config config;
    config.dataFormat = 2;
    auto handler = Leaks::MakeDataHandler(config, static_cast<DataType>(999));
    EXPECT_EQ(handler, nullptr);
}

TEST(DataHandler, DataHandler_Write_Type_False)
{
    g_isDlsymNullptr = false;
    std::shared_ptr<EventBase> data = std::make_shared<EventBase>();
    data->dataType = static_cast<DataType>(999);
    Config config;
    config.dataFormat = static_cast<uint8_t>(DataFormat::DB);
    strncpy_s(config.outputDir, sizeof(config.outputDir) - 1, "./testmsleaks", sizeof(config.outputDir) - 1);
    Utility::CreateDbPath(config, DB_DUMP_FILE);
    DbHandler handler(config, DataType::LEAKS_EVENT);
    EXPECT_FALSE(handler.Write(data));
    config.dataFormat = static_cast<uint8_t>(DataFormat::CSV);
    CsvHandler handler_(config, DataType::LEAKS_EVENT);
    EXPECT_FALSE(handler_.Write(data));
}

TEST(DataHandler, DataHandler_FixJson)
{
    std::string input = "\"{addr:20616937226752,size:28160,total:2097152,used:1617920}\"";
    std::string expected = R"({"addr":"20616937226752","size":"28160","total":"2097152","used":"1617920"})";
    EXPECT_EQ(FixJson(input), expected);
}