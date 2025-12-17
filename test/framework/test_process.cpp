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
#include <vector>
#include "securec.h"
#define private public
#include "process.h"
#undef private
#include "analysis/mstx_analyzer.h"
#include "client_parser.h"
#include "bit_field.h"
using namespace MemScope;

// 大部分单例config在此初始化，config相关修改请修改此处。
void setConfig(Config &config)
{
    BitField<decltype(config.eventType)> eventBit;
    BitField<decltype(config.levelType)> levelBit;
    BitField<decltype(config.analysisType)> analysisBit;
    analysisBit.setBit(static_cast<size_t>(AnalysisType::LEAKS_ANALYSIS));
    analysisBit.setBit(static_cast<size_t>(AnalysisType::DECOMPOSE_ANALYSIS));
    levelBit.setBit(static_cast<size_t>(LevelType::LEVEL_OP));
    levelBit.setBit(static_cast<size_t>(LevelType::LEVEL_KERNEL));
    eventBit.setBit(static_cast<size_t>(EventType::ALLOC_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::FREE_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::LAUNCH_EVENT));
    eventBit.setBit(static_cast<size_t>(EventType::ACCESS_EVENT));
    config.analysisType = analysisBit.getValue();
    config.eventType = eventBit.getValue();
    config.levelType = levelBit.getValue();
    config.enableCStack = true;
    config.enablePyStack = true;
    config.stepList.stepCount = 0;
    config.dataFormat = 0;
    config.collectMode = static_cast<uint8_t>(CollectMode::IMMEDIATE);
    strncpy_s(config.outputDir, sizeof(config.outputDir) - 1, "./testmsmemscope", sizeof(config.outputDir) - 1);
}

TEST(Process, process_setpreloadenv_without_atb_expect_success)
{
    unsetenv("ATB_HOME_PATH");
    setenv("LD_PRELOAD_PATH", "/lib64/", 1);
    Config config;
    Utility::FileCreateManager::GetInstance("testmsmemscope");
    Process process(config);
    process.SetPreloadEnv();
    char *env = getenv("LD_PRELOAD");
    std::string hooksSo = "libleaks_ascend_hal_hook.so:"
                          "libascend_mstx_hook.so:libascend_kernel_hook.so";
    EXPECT_EQ(std::string(env), hooksSo);
    setenv("LD_PRELOAD", "test.so", 1);
    process.SetPreloadEnv();
    env = getenv("LD_PRELOAD");
    EXPECT_EQ(std::string(env), hooksSo + ":test.so");
    unsetenv("LD_PRELOAD");
}

TEST(Process, process_setpreloadenv_with_atb_abi_0_expect_success)
{
    setenv("ATB_HOME_PATH", "/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_0", 1);
    setenv("LD_PRELOAD_PATH", "/lib64/", 1);
    Config config;
    Utility::FileCreateManager::GetInstance("testmsmemscope");
    Process process(config);
    process.SetPreloadEnv();
    char *env = getenv("LD_PRELOAD");
    std::string hooksSo = "libleaks_ascend_hal_hook.so:"
                          "libascend_mstx_hook.so:libascend_kernel_hook.so:libatb_abi_0_hook.so";
    EXPECT_EQ(std::string(env), hooksSo);
    setenv("LD_PRELOAD", "test.so", 1);
    process.SetPreloadEnv();
    env = getenv("LD_PRELOAD");
    EXPECT_EQ(std::string(env), hooksSo + ":test.so");
    unsetenv("LD_PRELOAD");
}

TEST(Process, process_setpreloadenv_with_atb_abi_1_expect_success)
{
    setenv("ATB_HOME_PATH", "/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1", 1);
    setenv("LD_PRELOAD_PATH", "/lib64/", 1);
    Config config;
    Utility::FileCreateManager::GetInstance("testmsmemscope");
    Process process(config);
    process.SetPreloadEnv();
    char *env = getenv("LD_PRELOAD");
    std::string hooksSo = "libleaks_ascend_hal_hook.so:"
                          "libascend_mstx_hook.so:libascend_kernel_hook.so:libatb_abi_1_hook.so";
    EXPECT_EQ(std::string(env), hooksSo);
    setenv("LD_PRELOAD", "test.so", 1);
    process.SetPreloadEnv();
    env = getenv("LD_PRELOAD");
    EXPECT_EQ(std::string(env), hooksSo + ":test.so");
    unsetenv("LD_PRELOAD");
}

TEST(Process, do_record_handler_except_success)
{
    auto buffer1 = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* record1 = buffer1.Cast<MemPoolRecord>();
    record1->type = RecordType::PTA_CACHING_POOL_RECORD;
    record1->recordIndex = 1;
    auto memoryusage1 = MemoryUsage {};
    memoryusage1.dataType = 0;
    memoryusage1.ptr = 12345;
    memoryusage1.allocSize = 512;
    memoryusage1.totalAllocated = 512;
    record1->memoryUsage = memoryusage1;

    auto buffer2 = RecordBuffer::CreateRecordBuffer<MstxRecord>();
    MstxRecord* record2 = buffer2.Cast<MstxRecord>();
    record2->type = RecordType::MSTX_MARK_RECORD;
    record2->markType = MarkType::RANGE_START_A;
    record2->rangeId = 0;
    record2->stepId = 1;
    record2->streamId = 123;

    auto buffer3 = RecordBuffer::CreateRecordBuffer<KernelLaunchRecord>();
    KernelLaunchRecord* record3 = buffer3.Cast<KernelLaunchRecord>();
    record3->type = RecordType::KERNEL_LAUNCH_RECORD;

    auto buffer4 = RecordBuffer::CreateRecordBuffer<AclItfRecord>();
    AclItfRecord* record4 = buffer4.Cast<AclItfRecord>();
    record4->type = RecordType::ACL_ITF_RECORD;

    auto buffer5 = RecordBuffer::CreateRecordBuffer<MemOpRecord>();
    MemOpRecord* record5 = buffer5.Cast<MemOpRecord>();
    record5->type = RecordType::MEMORY_RECORD;
    record5->subtype = RecordSubType::MALLOC;

    auto buffer6 = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* record6 = buffer6.Cast<MemPoolRecord>();
    record6->type = RecordType::PTA_WORKSPACE_POOL_RECORD;
    record6->recordIndex = 2;
    auto memoryusage6 = MemoryUsage {};
    memoryusage6.dataType = 0;
    memoryusage6.ptr = 12347;
    memoryusage6.allocSize = 512;
    memoryusage6.totalAllocated = 512;
    record6->memoryUsage = memoryusage6;

    Config config;
    setConfig(config);
    Process::GetInstance(config).RecordHandler(buffer1);
    Process::GetInstance(config).RecordHandler(buffer2);
    Process::GetInstance(config).RecordHandler(buffer3);
    Process::GetInstance(config).RecordHandler(buffer4);
    Process::GetInstance(config).RecordHandler(buffer5);
    Process::GetInstance(config).RecordHandler(buffer6);
}