// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <string>
#include "analyzer.h"
#include "record_info.h"
#include "config_info.h"

using namespace Leaks;

TEST(AnalyzerTest, AnalyzerConstruct) {
    AnalysisConfig analysisConfig;
    Analyzer analyzer(analysisConfig);
}

TEST(Analyzer, do_memory_record_expect_success)
{
    AnalysisConfig config;
    Analyzer analyzer(config);

    auto record = EventRecord{};
    record.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc = MemOpRecord {};
    memRecordMalloc.flag = 0xFF00000000000000;
    memRecordMalloc.modid = 99;
    memRecordMalloc.recordIndex = 123;
    memRecordMalloc.addr = 0x7958;
    memRecordMalloc.memSize = 1024;
    memRecordMalloc.timeStamp = 1234567;
    memRecordMalloc.memType = MemOpType::MALLOC;
    record.record.memoryRecord = memRecordMalloc;
    ClientId clientId = 0;
    analyzer.Do(clientId, record);
    
    memRecordMalloc.space = Leaks::MemOpSpace::HOST;
    memRecordMalloc.addr = 0x7959;
    record.record.memoryRecord = memRecordMalloc;
    analyzer.Do(clientId, record);

    auto memRecordFree = memRecordMalloc;
    memRecordFree.memType = MemOpType::FREE;
    memRecordFree.addr = 0x7958;
    record.record.memoryRecord = memRecordFree;

    analyzer.Do(clientId, record);

    memRecordFree.addr = 0x7959;
    record.record.memoryRecord = memRecordFree;

    analyzer.Do(clientId, record);
    
    analyzer.LeakAnalyze();
}

TEST(Analyzer, do_kernellaunch_record_expect_success)
{
    AnalysisConfig config;
    Analyzer analyzer(config);

    auto record = EventRecord{};
    record.type = RecordType::KERNEL_LAUNCH_RECORD;
    ClientId clientId = 0;
    analyzer.Do(clientId, record);

    auto record2 = EventRecord{};
    record2.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc = MemOpRecord {};
    memRecordMalloc.flag = 2377900603261207558;
    memRecordMalloc.recordIndex = 123;
    memRecordMalloc.addr = 0x7958;
    memRecordMalloc.memSize = 1024;
    memRecordMalloc.timeStamp = 1234567;
    memRecordMalloc.memType = MemOpType::MALLOC;
    record2.record.memoryRecord = memRecordMalloc;
    analyzer.Do(clientId, record2);
}

TEST(Analyzer, do_aclitf_record_expect_success)
{
    AnalysisConfig config;
    Analyzer analyzer(config);

    auto record = EventRecord{};
    record.type = RecordType::ACL_ITF_RECORD;
    ClientId clientId = 0;
    analyzer.Do(clientId, record);

    auto record2 = EventRecord{};
    record2.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc = MemOpRecord {};
    memRecordMalloc.flag = 2377900603261207558;
    memRecordMalloc.recordIndex = 123;
    memRecordMalloc.addr = 0x7958;
    memRecordMalloc.memSize = 1024;
    memRecordMalloc.timeStamp = 1234567;
    memRecordMalloc.memType = MemOpType::MALLOC;
    record2.record.memoryRecord = memRecordMalloc;
    analyzer.Do(clientId, record2);
}

TEST(AnalyzerTest, AnalyzerLeakAnalyze) {
    AnalysisConfig config;
    Analyzer analyzer(config);

    ClientId clientId = 0;
    auto record = EventRecord{};
    record.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc = MemOpRecord {};
    memRecordMalloc.flag = 2377900603261207558;
    memRecordMalloc.recordIndex = 123;
    memRecordMalloc.addr = 0x7958;
    memRecordMalloc.memSize = 1024;
    memRecordMalloc.timeStamp = 1234567;
    memRecordMalloc.memType = MemOpType::MALLOC;
    record.record.memoryRecord = memRecordMalloc;

    analyzer.Do(clientId, record);
    analyzer.LeakAnalyze();
}

TEST(MemoryHashTableTest, do_record_leaks) {
    AnalysisConfig config;
    Analyzer analyzer(config);

    ClientId clientId = 0;
    auto record1 = EventRecord{};
    record1.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc1 = MemOpRecord {};
    memRecordMalloc1.flag = 2377900603261207558;
    memRecordMalloc1.recordIndex = 1;
    memRecordMalloc1.space = MemOpSpace::DEVICE;
    memRecordMalloc1.memType = MemOpType::MALLOC;
    memRecordMalloc1.addr = 0x7958;
    memRecordMalloc1.memSize = 1024;
    memRecordMalloc1.timeStamp = 1234567;
    record1.record.memoryRecord = memRecordMalloc1;

    auto record2 = EventRecord{};
    record2.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc2 = MemOpRecord {};
    memRecordMalloc2.flag = 18374686480754951175;
    memRecordMalloc2.recordIndex = 2;
    memRecordMalloc2.space = MemOpSpace::INVALID;
    memRecordMalloc2.memType = MemOpType::MALLOC;
    memRecordMalloc2.addr = 0x7957;
    memRecordMalloc2.memSize = 512;
    memRecordMalloc2.timeStamp = 1234568;
    record2.record.memoryRecord = memRecordMalloc2;

    auto record4 = EventRecord{};
    record4.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc4 = MemOpRecord {};
    memRecordMalloc4.flag = 504403158275081222;
    memRecordMalloc4.recordIndex = 4;
    memRecordMalloc4.space = MemOpSpace::DEVICE;
    memRecordMalloc4.memType = MemOpType::MALLOC;
    memRecordMalloc4.addr = 0x7960;
    memRecordMalloc4.memSize = 1024;
    memRecordMalloc4.timeStamp = 1234557;
    record4.record.memoryRecord = memRecordMalloc4;

    analyzer.Do(clientId, record1);
    analyzer.Do(clientId, record2);
    analyzer.Do(clientId, record4);
    analyzer.LeakAnalyze();
}

TEST(MemoryHashTableTest, do_record_no_leaks) {
    AnalysisConfig config;
    Analyzer analyzer(config);

    ClientId clientId = 0;
    auto record1 = EventRecord{};
    record1.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc1 = MemOpRecord {};
    memRecordMalloc1.flag = 2377900603261207558;
    memRecordMalloc1.recordIndex = 1;
    memRecordMalloc1.space = MemOpSpace::DEVICE;
    memRecordMalloc1.memType = MemOpType::MALLOC;
    memRecordMalloc1.addr = 0x7958;
    memRecordMalloc1.memSize = 1024;
    memRecordMalloc1.timeStamp = 1234567;
    record1.record.memoryRecord = memRecordMalloc1;

    auto record3 = EventRecord{};
    record3.type = RecordType::MEMORY_RECORD;
    auto memRecordFree = MemOpRecord {};
    memRecordFree.recordIndex = 3;
    memRecordFree.space = MemOpSpace::INVALID;
    memRecordFree.memType = MemOpType::FREE;
    memRecordFree.addr = 0x7958;
    memRecordFree.memSize = 0;
    record3.record.memoryRecord = memRecordFree;

    analyzer.Do(clientId, record1);
    analyzer.Do(clientId, record3);
    analyzer.LeakAnalyze();
}

TEST(MemoryHashTableTest, do_record_double_free) {
    AnalysisConfig config;
    Analyzer analyzer(config);

    ClientId clientId = 0;
    auto record1 = EventRecord{};
    record1.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc1 = MemOpRecord {};
    memRecordMalloc1.flag = 2377900603261207558;
    memRecordMalloc1.recordIndex = 1;
    memRecordMalloc1.space = MemOpSpace::DEVICE;
    memRecordMalloc1.memType = MemOpType::MALLOC;
    memRecordMalloc1.addr = 0x7958;
    memRecordMalloc1.memSize = 1024;
    memRecordMalloc1.timeStamp = 1234567;
    record1.record.memoryRecord = memRecordMalloc1;

    auto record2 = EventRecord{};
    record2.type = RecordType::MEMORY_RECORD;
    auto memRecordFree1 = MemOpRecord {};
    memRecordFree1.recordIndex = 2;
    memRecordFree1.space = MemOpSpace::INVALID;
    memRecordFree1.memType = MemOpType::FREE;
    memRecordFree1.addr = 0x7958;
    memRecordFree1.memSize = 0;
    record2.record.memoryRecord = memRecordFree1;

    auto record3 = EventRecord{};
    record3.type = RecordType::MEMORY_RECORD;
    auto memRecordFree2 = MemOpRecord {};
    memRecordFree2.recordIndex = 3;
    memRecordFree2.space = MemOpSpace::INVALID;
    memRecordFree2.memType = MemOpType::FREE;
    memRecordFree2.addr = 0x7958;
    memRecordFree2.memSize = 0;
    record3.record.memoryRecord = memRecordFree2;

    analyzer.Do(clientId, record1);
    analyzer.Do(clientId, record2);
    analyzer.Do(clientId, record3);
    analyzer.LeakAnalyze();
}

TEST(MemoryHashTableTest, do_record_double_malloc) {
    AnalysisConfig config;
    Analyzer analyzer(config);

    ClientId clientId = 0;
    auto record1 = EventRecord{};
    record1.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc1 = MemOpRecord {};
    memRecordMalloc1.flag = 2377900603261207558;
    memRecordMalloc1.recordIndex = 1;
    memRecordMalloc1.space = MemOpSpace::DEVICE;
    memRecordMalloc1.memType = MemOpType::MALLOC;
    memRecordMalloc1.addr = 0x7958;
    memRecordMalloc1.memSize = 1024;
    memRecordMalloc1.timeStamp = 1234567;
    record1.record.memoryRecord = memRecordMalloc1;

    auto record2 = EventRecord{};
    record2.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc2 = MemOpRecord {};
    memRecordMalloc2.flag = 2377900603261207558;
    memRecordMalloc2.recordIndex = 2;
    memRecordMalloc2.space = MemOpSpace::DEVICE;
    memRecordMalloc2.memType = MemOpType::MALLOC;
    memRecordMalloc2.addr = 0x7958;
    memRecordMalloc2.memSize = 1024;
    memRecordMalloc2.timeStamp = 1234567;
    record2.record.memoryRecord = memRecordMalloc2;

    analyzer.Do(clientId, record1);
    analyzer.Do(clientId, record2);
    analyzer.LeakAnalyze();
}

TEST(MemoryHashTableTest, do_record_free_null) {
    AnalysisConfig config;
    Analyzer analyzer(config);

    ClientId clientId = 0;
    auto record1 = EventRecord{};
    record1.type = RecordType::MEMORY_RECORD;
    auto memRecordFree1 = MemOpRecord {};
    memRecordFree1.recordIndex = 1;
    memRecordFree1.space = MemOpSpace::INVALID;
    memRecordFree1.memType = MemOpType::FREE;
    memRecordFree1.addr = 0x7958;
    memRecordFree1.memSize = 0;
    record1.record.memoryRecord = memRecordFree1;

    analyzer.Do(clientId, record1);
    analyzer.LeakAnalyze();
}

TEST(MemoryHashTableTest, do_record_fail) {
    AnalysisConfig config;
    Analyzer analyzer(config);

    ClientId clientId = 0;
    auto record1 = EventRecord{};
    record1.type = RecordType::MEMORY_RECORD;
    auto memRecordFree1 = MemOpRecord {};
    memRecordFree1.recordIndex = 1;
    memRecordFree1.space = MemOpSpace::INVALID;
    memRecordFree1.memType = MemOpType::FREE;
    memRecordFree1.addr = 0x7958;
    memRecordFree1.memSize = 0;
    record1.record.memoryRecord = memRecordFree1;

    analyzer.Do(clientId, record1);
    analyzer.LeakAnalyze();
}

TEST(MemoryHashTableTest, do_memory_record_nulltable) {
    AnalysisConfig config;
    Analyzer analyzer(config);

    auto record = EventRecord{};
    record.type = RecordType::MEMORY_RECORD;
    auto memRecordFree = MemOpRecord {};
    memRecordFree.recordIndex = 123;
    memRecordFree.addr = 0x7958;
    ClientId clientId = 0;
    memRecordFree.memType = MemOpType::FREE;
    record.record.memoryRecord = memRecordFree;
    analyzer.Do(clientId, record);

    analyzer.LeakAnalyze();
}

TEST(TorchnputraceTest, do_npu_trace_record_success)
{
    AnalysisConfig config;
    Analyzer analyzer(config);

    auto record = EventRecord{};
    record.type = RecordType::TORCH_NPU_RECORD;
    TorchNpuRecord torchNpuRecord;
    MemoryUsage memoryUsage;
    torchNpuRecord.memoryUsage = memoryUsage;
    record.record.torchNpuRecord = torchNpuRecord;
    ClientId clientId = 0;
    analyzer.Do(clientId, record);
}