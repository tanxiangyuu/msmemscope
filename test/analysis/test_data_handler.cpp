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
    strncpy_s(config.outputDir, sizeof(config.outputDir) - 1, "./testLeaksDumpResults", sizeof(config.outputDir) - 1);

    CsvHandler handler(config, DumpClass::LEAKS_RECORD);
    handler.Init();
    DumpContainer data;
    data.id = 1;
    data.event = "memory_leak";
    data.eventType = "type1";
    data.name = "obj1";
    data.timeStamp = 123456789;
    data.pid = 1234;
    data.tid = 5678;
    data.deviceId = "dev1";
    data.addr = "0x1234";
    data.attr = "size=1024";
    data.dumpType = DumpClass::LEAKS_RECORD;
    CallStackString stack = {};
    ASSERT_TRUE(handler.Write(&data, stack));

    config.enablePyStack = true;
    CsvHandler handlerPy(config, DumpClass::LEAKS_RECORD);
    handlerPy.Init();
    CallStackString stack_;
    stack_.pyStack = "call_stack_py";
    ASSERT_TRUE(handlerPy.Write(&data, stack_));

    CsvHandler handler_(config, DumpClass::PYTHON_TRACE);
    handler_.Init();
    TraceEvent event;
    event.startTs = 1000;
    event.endTs = 2000;
    event.tid = 123;
    event.pid = 456;
    event.info = "function_call";
    event.hash = "hash123";
    event.dumpType = DumpClass::PYTHON_TRACE;
    ASSERT_TRUE(handler_.Write(&event, {}));
}

TEST(DataHandler, Sqlite3_open)
{
    g_isDlsymNullptr = false;
    sqlite3* db = nullptr;
    std::string path = "./testLeaksDumpResults/test.db";
    int rc = Sqlite3Open(path.c_str(), &db);
    EXPECT_EQ(rc, 0);
}

TEST(DataHandler, DbHandler_Write_LeakRecord)
{
    g_isDlsymNullptr = false;
    Config config;
    config.dataFormat = static_cast<uint8_t>(DataFormat::DB);
    config.enableCStack = true;
    config.enablePyStack = true;
    strncpy_s(config.outputDir, sizeof(config.outputDir) - 1, "./testLeaksDumpResults", sizeof(config.outputDir) - 1);
    Utility::CreateDbPath(config, DB_DUMP_FILE);

    std::unique_ptr<DataHandler> handler = MakeDataHandler(config, DumpClass::LEAKS_RECORD);
    handler->Init();
    DumpContainer data;
    data.id = 1;
    data.event = "memory_leak";
    data.eventType = "type1";
    data.name = "obj1";
    data.timeStamp = 123456789;
    data.pid = 1234;
    data.tid = 5678;
    data.deviceId = "dev1";
    data.addr = "0x1234";
    data.attr = "size=1024";
    data.dumpType = DumpClass::LEAKS_RECORD;
    CallStackString stack;
    stack.cStack = "call_stack_c";
    stack.pyStack = "call_stack_py";
    ASSERT_TRUE(handler->Write(&data, stack));

    config.enableCStack = false;
    config.enablePyStack = false;
    DbHandler handler_(config, DumpClass::PYTHON_TRACE);
    handler_.Init();
    TraceEvent event;
    event.startTs = 1000;
    event.endTs = 2000;
    event.tid = 123;
    event.pid = 456;
    event.info = "function_call";
    event.hash = "hash123";
    event.dumpType = DumpClass::PYTHON_TRACE;
    ASSERT_TRUE(handler_.Write(&event, {}));
}

TEST(DataHandler, CsvHandler_InitSetParm_Default)
{
    Config config;
    config.dataFormat = static_cast<uint8_t>(DataFormat::CSV);
    strncpy_s(config.outputDir, sizeof(config.outputDir) - 1, "./testLeaksDumpResults", sizeof(config.outputDir) - 1);
    CsvHandler handler(config, static_cast<DumpClass>(999));
    EXPECT_TRUE(true);
}

TEST(DataHandler, CsvHandler_Write_NullData)
{
    Config config;
    config.dataFormat = static_cast<uint8_t>(DataFormat::CSV);
    strncpy_s(config.outputDir, sizeof(config.outputDir) - 1, "./testLeaksDumpResults", sizeof(config.outputDir) - 1);
    CsvHandler handler(config, DumpClass::LEAKS_RECORD);
    handler.Init();
    ASSERT_FALSE(handler.Write(nullptr, {}));
}

TEST(DataHandler, DbHandler_InitSetParm_Default)
{
    Config config;
    config.dataFormat = static_cast<uint8_t>(DataFormat::DB);
    strncpy_s(config.outputDir, sizeof(config.outputDir) - 1, "./testLeaksDumpResults", sizeof(config.outputDir) - 1);
    DbHandler handler(config, static_cast<DumpClass>(999));
    EXPECT_TRUE(true);
}

TEST(DataHandler, DbHandler_Write_NullData)
{
    g_isDlsymNullptr = false;
    Config config;
    config.dataFormat = static_cast<uint8_t>(DataFormat::DB);
    config.enableCStack = false;
    config.enablePyStack = false;
    strncpy_s(config.outputDir, sizeof(config.outputDir) - 1, "./testLeaksDumpResults", sizeof(config.outputDir) - 1);
    Utility::CreateDbPath(config, DB_DUMP_FILE);
    DbHandler handler(config, DumpClass::LEAKS_RECORD);
    handler.Init();
    ASSERT_FALSE(handler.Write(nullptr, {}));
}

TEST(DataHandler, MakeDataHandler_FALSE)
{
    Config config;
    config.dataFormat = 2;
    DumpClass data = static_cast<DumpClass>(999);
    auto handler = Leaks::MakeDataHandler(config, data);
    EXPECT_EQ(handler, nullptr);
}

TEST(DataHandler, write_false_type)
{
    g_isDlsymNullptr = false;
    DumpDataClass data(static_cast<DumpClass>(2));
    CallStackString stack = {};
    Config config;
    config.dataFormat = static_cast<uint8_t>(DataFormat::DB);
    DbHandler handler(config, DumpClass::LEAKS_RECORD);
    EXPECT_FALSE(handler.Write(&data, stack));
    config.dataFormat = static_cast<uint8_t>(DataFormat::CSV);
    CsvHandler handler_(config, DumpClass::LEAKS_RECORD);
    EXPECT_FALSE(handler_.Write(&data, stack));
}

TEST(DataHandler, fixjson)
{
    std::string input = "\"{addr:20616937226752,size:28160,total:2097152,used:1617920}\"";
    std::string expected = R"({"addr":"20616937226752","size":"28160","total":"2097152","used":"1617920"})";
    EXPECT_EQ(FixJson(input), expected);
}