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
#include <unordered_map>
#include "python_trace.h"

using namespace MemScope;

class PythonTraceTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        Utility::FileCreateManager::GetInstance("./testmsmemscope").SetProjectDir("./testmsmemscope");
    }

    void TearDown() override
    {
        Utility::FileCreateManager::GetInstance("./testmsmemscope").SetProjectDir("");
        rmdir("./testmsmemscope");
    }
};

TEST_F(PythonTraceTest, RecordPyCallTest)
{
    PythonTrace::GetInstance().RecordPyCall("123", "123", 0);
    PythonTrace::GetInstance().RecordPyCall("123:__torch_dispatch__", "123", 0);
}

TEST_F(PythonTraceTest, RecordCCallTest)
{
    PythonTrace::GetInstance().RecordCCall("123", "123");
    PythonTrace::GetInstance().RecordReturn("123:__torch_dispatch__", "123");
}

TEST_F(PythonTraceTest, RecordReturnTest)
{
    PythonTrace::GetInstance().Start();
    PythonTrace::GetInstance().RecordReturn("123", "123");
    PythonTrace::GetInstance().RecordPyCall("123", "123", 0);
    PythonTrace::GetInstance().RecordReturn("123", "123");

    std::string hash;
    std::string info;
    uint64_t timestamp = 0;
    PyTraceType what = PyTraceType::PYCALL;
    callback(hash, info, what, timestamp);
    what = PyTraceType::PYRETURN;
    callback(hash, info, what, timestamp);
    what = PyTraceType::CCALL;
    callback(hash, info, what, timestamp);
    what = PyTraceType::CRETURN;
    callback(hash, info, what, timestamp);
    PythonTrace::GetInstance().Stop();
}