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
#define private public
#include "event_trace/memory_watch/tensor_dumper.h"
#undef private
#include "event_trace/memory_watch/memory_watch.h"
#include "event_trace/memory_watch/tensor_monitor.h"
#include "file.h"

using namespace MemScope;

TEST(TesnorDumperTest, dump_tensor_binary_expect_success)
{
    TensorDumper::GetInstance().dumpDir_ = "./testmsmemscope/watch_dump";
    Utility::MakeDir(TensorDumper::GetInstance().dumpDir_);
    std::string fileName = "test";
    std::vector<uint8_t> hostData(fileName.begin(), fileName.end());
    auto ret = TensorDumper::GetInstance().DumpTensorBinary(hostData, fileName);
    ASSERT_TRUE(ret);
}

TEST(TesnorDumperTest, dump_expect_success)
{
    std::string fileName = "test";
    MonitoredTensor tensor = {};
    uint64_t ptr = 6666;
    tensor.data = reinterpret_cast<void *>(ptr);
    tensor.dataSize = 100;
    TensorDumper::GetInstance().SetDumpName(ptr, "test_op");
    TensorDumper::GetInstance().SetDumpNums(ptr, 2);
    TensorMonitor::GetInstance().pythonWatchedTensorsMap_.insert({ptr, tensor});
    TensorDumper::GetInstance().Dump(nullptr, fileName, true);
    EXPECT_EQ(TensorDumper::GetInstance().GetDumpNums(ptr), 1);

    TensorDumper::GetInstance().SetDumpNums(ptr, 0);
    TensorDumper::GetInstance().Dump(nullptr, fileName, true);
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