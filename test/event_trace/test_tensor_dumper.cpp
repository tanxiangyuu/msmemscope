// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include <gtest/gtest.h>
#define private public
#include "event_trace/op_watch/tensor_dumper.h"
#undef private
#include "event_trace/op_watch/op_excute_watch.h"
#include "event_trace/op_watch/tensor_monitor.h"

using namespace Leaks;

TEST(TesnorDumperTest, dump_tensor_binary_expect_success)
{
    std::string fileName = "test";
    std::vector<uint8_t> hostData(fileName.begin(), fileName.end());
    auto ret = TensorDumper::GetInstance().DumpTensorBinary(hostData, fileName);
    ASSERT_TRUE(ret);
}

TEST(TesnorDumperTest, dump_tensor_MD5_expect_success)
{
    std::string fileName = "test";
    std::vector<uint8_t> hostData(fileName.begin(), fileName.end());
    auto ret = TensorDumper::GetInstance().DumpTensorHashValue(hostData, fileName);
    ASSERT_TRUE(ret);
}

TEST(TesnorDumperTest, dump_expect_success)
{
    OpEventType type = OpEventType::ATEN_START;
    std::string fileName = "test";
    MonitoredTensor tensor = {};
    uint64_t ptr = 6666;
    tensor.data = reinterpret_cast<void *>(ptr);
    tensor.dataSize = 100;
    TensorDumper::GetInstance().SetDumpName(ptr, "test_op");
    TensorDumper::GetInstance().SetDumpNums(ptr, 2);
    TensorMonitor::GetInstance().pythonWatchedTensorsMap_.insert({ptr, tensor});
    TensorDumper::GetInstance().Dump(nullptr, fileName, type);
    EXPECT_EQ(TensorDumper::GetInstance().GetDumpNums(ptr), 1);

    TensorDumper::GetInstance().SetDumpNums(ptr, 0);
    TensorDumper::GetInstance().Dump(nullptr, fileName, type);
}

TEST(TesnorDumperTest, set_dump_name_expect_success)
{
    std::string name = "test";
    uint64_t ptr = 6666;
    TensorDumper::GetInstance().SetDumpName(ptr, name);
    EXPECT_EQ(TensorDumper::GetInstance().GetDumpName(ptr), name);
}

TEST(TesnorDumperTest, delete_dump_name_expect_success)
{
    std::string name = "test";
    uint64_t ptr = 6666;
    TensorDumper::GetInstance().SetDumpName(ptr, name);
    TensorDumper::GetInstance().DeleteDumpName(ptr);
    EXPECT_EQ(TensorDumper::GetInstance().GetDumpName(ptr), "UNKNWON");
}

TEST(TesnorDumperTest, set_dump_nums_expect_success)
{
    int32_t dumpNums = 100;
    uint64_t ptr = 6666;
    TensorDumper::GetInstance().SetDumpNums(ptr, dumpNums);
    EXPECT_EQ(TensorDumper::GetInstance().GetDumpNums(ptr), dumpNums);
}

TEST(TesnorDumperTest, delete_dump_nums_expect_success)
{
    int32_t dumpNums = 100;
    uint64_t ptr = 6666;
    TensorDumper::GetInstance().SetDumpNums(ptr, dumpNums);
    TensorDumper::GetInstance().DeleteDumpNums(ptr);
    EXPECT_EQ(TensorDumper::GetInstance().GetDumpNums(ptr), -1);
}