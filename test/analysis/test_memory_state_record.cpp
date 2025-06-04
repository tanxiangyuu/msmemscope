// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#define private public
#include "memory_state_record.h"
#undef private
#include "securec.h"

using namespace Leaks;

TEST(MemoryStateRecordTest, state_cpu_memory_record_expect_success)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::MEMORY_RECORD;
    auto memRecordMalloc = MemOpRecord{};
    memRecordMalloc.devType = DeviceType::CPU;
    memRecordMalloc.tid = 10;
    memRecordMalloc.pid = 10;
    memRecordMalloc.flag = 10;
    memRecordMalloc.modid = 55;
    memRecordMalloc.devId = 10;
    memRecordMalloc.recordIndex = 102;
    memRecordMalloc.kernelIndex = 101;
    memRecordMalloc.space = MemOpSpace::HOST;
    memRecordMalloc.addr = 0x1234;
    memRecordMalloc.memSize = 128;
    memRecordMalloc.timeStamp = 789;
    memRecordMalloc.memType = MemOpType::MALLOC;
    record.eventRecord.record.memoryRecord = memRecordMalloc;

    CallStackString stack{};
    Config config;
    MemoryStateRecord memoryStateRecord{config};
    memoryStateRecord.MemoryInfoProcess(record, stack);
    auto key = std::make_pair("common", memRecordMalloc.addr);
    ASSERT_EQ(memoryStateRecord.ptrMemoryInfoMap_.size(), 1);
    ASSERT_EQ(memoryStateRecord.ptrMemoryInfoMap_[key].size(), 1);

    record.eventRecord.record.memoryRecord.memType = MemOpType::FREE;
    memoryStateRecord.MemoryInfoProcess(record, stack);
    ASSERT_EQ(memoryStateRecord.ptrMemoryInfoMap_.size(), 1);
    ASSERT_EQ(memoryStateRecord.ptrMemoryInfoMap_[key].size(), 2);
}

TEST(MemoryStateRecordTest, state_ATB_memory_pool_record_expect_success)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::ATB_MEMORY_POOL_RECORD;
    auto atbPoolRecord = MemPoolRecord{};
    atbPoolRecord.recordIndex = 1;
    atbPoolRecord.pid = 1234;
    atbPoolRecord.tid = 1234;
    atbPoolRecord.timeStamp = 1000;
    atbPoolRecord.devId = 1;
    auto atbMemUsage = MemoryUsage{};
    atbMemUsage.ptr = 1234;
    atbMemUsage.allocSize = 100;
    atbMemUsage.totalAllocated = 10000;
    atbMemUsage.totalReserved = 30000;
    atbMemUsage.totalActive = 10000;
    atbPoolRecord.memoryUsage = atbMemUsage;
    record.eventRecord.record.memPoolRecord = atbPoolRecord;

    CallStackString stack{};
    Config config;
    MemoryStateRecord memoryStateRecord{config};
    auto key = std::make_pair("ATB", atbPoolRecord.memoryUsage.ptr);
    memoryStateRecord.MemoryPoolInfoProcess(record, stack);
    ASSERT_EQ(memoryStateRecord.ptrMemoryInfoMap_.size(), 1);
    ASSERT_EQ(memoryStateRecord.ptrMemoryInfoMap_[key].size(), 1);

    atbPoolRecord.memoryUsage.allocSize = -100;
    memoryStateRecord.MemoryPoolInfoProcess(record, stack);
    ASSERT_EQ(memoryStateRecord.ptrMemoryInfoMap_.size(), 1);
    ASSERT_EQ(memoryStateRecord.ptrMemoryInfoMap_[key].size(), 2);
}

TEST(MemoryStateRecordTest, state_PTA_memory_pool_record_with_decompose_expect_success)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::TORCH_NPU_RECORD;
    auto ptaPoolRecord = MemPoolRecord{};
    ptaPoolRecord.recordIndex = 1;
    ptaPoolRecord.pid = 1234;
    ptaPoolRecord.tid = 1234;
    ptaPoolRecord.timeStamp = 1000;
    ptaPoolRecord.devId = 1;
    auto ptaMemUsage = MemoryUsage{};
    ptaMemUsage.ptr = 1234;
    ptaMemUsage.allocSize = 100;
    ptaMemUsage.totalAllocated = 10000;
    ptaMemUsage.totalReserved = 30000;
    ptaPoolRecord.memoryUsage = ptaMemUsage;
    record.eventRecord.record.memPoolRecord = ptaPoolRecord;

    CallStackString stack{};
    Config config;
    config.analysisType = 2;
    MemoryStateRecord memoryStateRecord{config};
    auto key = std::make_pair("PTA", ptaPoolRecord.memoryUsage.ptr);
    memoryStateRecord.MemoryPoolInfoProcess(record, stack);
    ASSERT_EQ(memoryStateRecord.ptrMemoryInfoMap_.size(), 1);
    ASSERT_EQ(memoryStateRecord.ptrMemoryInfoMap_[key].size(), 1);

    ptaPoolRecord.memoryUsage.allocSize = -100;
    memoryStateRecord.MemoryPoolInfoProcess(record, stack);
    ASSERT_EQ(memoryStateRecord.ptrMemoryInfoMap_.size(), 1);
    ASSERT_EQ(memoryStateRecord.ptrMemoryInfoMap_[key].size(), 2);
}

TEST(MemoryStateRecordTest, state_mindspore_memory_pool_record_with_decompose_expect_success)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::MINDSPORE_NPU_RECORD;
    auto msPoolRecord = MemPoolRecord{};
    msPoolRecord.recordIndex = 1;
    msPoolRecord.pid = 1234;
    msPoolRecord.tid = 1234;
    msPoolRecord.timeStamp = 1000;
    msPoolRecord.devId = 1;
    auto msMemUsage = MemoryUsage{};
    msMemUsage.ptr = 1234;
    msMemUsage.allocSize = 100;
    msMemUsage.totalAllocated = 10000;
    msMemUsage.totalReserved = 30000;
    msPoolRecord.memoryUsage = msMemUsage;
    record.eventRecord.record.memPoolRecord = msPoolRecord;

    CallStackString stack{};
    Config config;
    config.analysisType = 2;
    MemoryStateRecord memoryStateRecord{config};
    auto key = std::make_pair("MINDSPORE", msPoolRecord.memoryUsage.ptr);
    memoryStateRecord.MemoryPoolInfoProcess(record, stack);
    ASSERT_EQ(memoryStateRecord.ptrMemoryInfoMap_.size(), 1);
    ASSERT_EQ(memoryStateRecord.ptrMemoryInfoMap_[key].size(), 1);

    msPoolRecord.memoryUsage.allocSize = -100;
    memoryStateRecord.MemoryPoolInfoProcess(record, stack);
    ASSERT_EQ(memoryStateRecord.ptrMemoryInfoMap_.size(), 1);
    ASSERT_EQ(memoryStateRecord.ptrMemoryInfoMap_[key].size(), 2);
}

TEST(MemoryStateRecordTest, state_unknown_memory_pool_record)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::INVALID_RECORD;

    CallStackString stack{};
    Config config;
    MemoryStateRecord memoryStateRecord{config};
    memoryStateRecord.MemoryPoolInfoProcess(record, stack);
}

TEST(MemoryStateRecordTest, state_addr_info_record_expect_success)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::ADDR_INFO_RECORD;
    auto info = AddrInfo{};
    info.addr = 123;
    record.eventRecord.record.addrInfo = info;
    CallStackString stack{};
    Config config;
    MemoryStateRecord memoryStateRecord{config};
    memoryStateRecord.MemoryAddrInfoProcess(record, stack);

    record.eventRecord.type = RecordType::TORCH_NPU_RECORD;
    auto memPoolRecord = MemPoolRecord{};
    memPoolRecord.type = RecordType::TORCH_NPU_RECORD;
    memPoolRecord.memoryUsage.ptr = 123;
    memPoolRecord.recordIndex = 1;
    memPoolRecord.pid = 1234;
    memPoolRecord.tid = 1234;
    memPoolRecord.timeStamp = 1000;
    memPoolRecord.devId = 1;
    auto ptaMemUsage = MemoryUsage{};
    ptaMemUsage.ptr = 123;
    ptaMemUsage.allocSize = 100;
    ptaMemUsage.totalAllocated = 10000;
    ptaMemUsage.totalReserved = 30000;
    ptaMemUsage.totalActive = 10000;
    memPoolRecord.memoryUsage = ptaMemUsage;
    record.eventRecord.record.memPoolRecord = memPoolRecord;

    memoryStateRecord.MemoryPoolInfoProcess(record, stack);

    record.eventRecord.type = RecordType::ADDR_INFO_RECORD;
    record.eventRecord.record.addrInfo = info;
    memoryStateRecord.MemoryAddrInfoProcess(record, stack);
}

TEST(MemoryStateRecordTest, state_addr_info_record_set_config)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::ADDR_INFO_RECORD;
    auto info = AddrInfo{};
    info.addr = 123;
    record.eventRecord.record.addrInfo = info;
    CallStackString stack{};
    Config config;
    BitField<decltype(config.analysisType)> analysisBit;
    analysisBit.setBit(static_cast<size_t>(AnalysisType::DECOMPOSE_ANALYSIS));
    config.analysisType = analysisBit.getValue();
    MemoryStateRecord memoryStateRecord{config};
    memoryStateRecord.MemoryAddrInfoProcess(record, stack);

    record.eventRecord.type = RecordType::TORCH_NPU_RECORD;
    auto memPoolRecord = MemPoolRecord{};
    memPoolRecord.type = RecordType::TORCH_NPU_RECORD;
    memPoolRecord.memoryUsage.ptr = 123;
    memPoolRecord.recordIndex = 1;
    memPoolRecord.pid = 1234;
    memPoolRecord.tid = 1234;
    memPoolRecord.timeStamp = 1000;
    memPoolRecord.devId = 1;
    auto ptaMemUsage = MemoryUsage{};
    ptaMemUsage.ptr = 123;
    ptaMemUsage.allocSize = 100;
    ptaMemUsage.totalAllocated = 10000;
    ptaMemUsage.totalReserved = 30000;
    ptaMemUsage.totalActive = 10000;
    memPoolRecord.memoryUsage = ptaMemUsage;
    record.eventRecord.record.memPoolRecord = memPoolRecord;

    memoryStateRecord.MemoryPoolInfoProcess(record, stack);

    record.eventRecord.type = RecordType::ADDR_INFO_RECORD;
    record.eventRecord.record.addrInfo = info;
    memoryStateRecord.MemoryAddrInfoProcess(record, stack);
}

TEST(MemoryStateRecordTest, state_addr_info_record_for_malloc)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::ADDR_INFO_RECORD;
    auto info = AddrInfo{};
    info.addr = 123;
    info.type = AddrInfoType::PTA_OPTIMIZER_STEP;
    strncpy_s(info.owner, sizeof(info.owner), "weight", sizeof(info.owner) - 1);
    record.eventRecord.record.addrInfo = info;
    CallStackString stack{};
    Config config;
    BitField<decltype(config.analysisType)> analysisBit;
    analysisBit.setBit(static_cast<size_t>(AnalysisType::DECOMPOSE_ANALYSIS));
    config.analysisType = analysisBit.getValue();
    MemoryStateRecord memoryStateRecord{config};
    memoryStateRecord.MemoryAddrInfoProcess(record, stack);

    record.eventRecord.type = RecordType::TORCH_NPU_RECORD;
    auto memPoolRecord = MemPoolRecord{};
    memPoolRecord.type = RecordType::TORCH_NPU_RECORD;
    memPoolRecord.memoryUsage.ptr = 123;
    memPoolRecord.recordIndex = 1;
    memPoolRecord.pid = 1234;
    memPoolRecord.tid = 1234;
    memPoolRecord.timeStamp = 1000;
    memPoolRecord.devId = 1;
    auto ptaMemUsage = MemoryUsage{};
    ptaMemUsage.ptr = 123;
    ptaMemUsage.allocSize = 100;
    memPoolRecord.memoryUsage = ptaMemUsage;
    record.eventRecord.record.memPoolRecord = memPoolRecord;

    memoryStateRecord.MemoryPoolInfoProcess(record, stack);

    record.eventRecord.type = RecordType::ADDR_INFO_RECORD;
    record.eventRecord.record.addrInfo = info;
    memoryStateRecord.MemoryAddrInfoProcess(record, stack);
}

TEST(MemoryStateRecordTest, state_addr_info_record_for_free)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::ADDR_INFO_RECORD;
    auto info = AddrInfo{};
    info.addr = 123;
    info.type = AddrInfoType::PTA_OPTIMIZER_STEP;
    strncpy_s(info.owner, sizeof(info.owner), "weight", sizeof(info.owner) - 1);
    record.eventRecord.record.addrInfo = info;
    CallStackString stack{};
    Config config;
    BitField<decltype(config.analysisType)> analysisBit;
    analysisBit.setBit(static_cast<size_t>(AnalysisType::DECOMPOSE_ANALYSIS));
    config.analysisType = analysisBit.getValue();
    MemoryStateRecord memoryStateRecord{config};
    memoryStateRecord.MemoryAddrInfoProcess(record, stack);

    record.eventRecord.type = RecordType::TORCH_NPU_RECORD;
    auto memPoolRecord = MemPoolRecord{};
    memPoolRecord.type = RecordType::TORCH_NPU_RECORD;
    memPoolRecord.memoryUsage.ptr = 123;
    memPoolRecord.recordIndex = 1;
    memPoolRecord.pid = 1234;
    memPoolRecord.tid = 1234;
    memPoolRecord.timeStamp = 1000;
    memPoolRecord.devId = 1;
    auto ptaMemUsage = MemoryUsage{};
    ptaMemUsage.ptr = 123;
    ptaMemUsage.allocSize = -100;
    memPoolRecord.memoryUsage = ptaMemUsage;
    record.eventRecord.record.memPoolRecord = memPoolRecord;

    memoryStateRecord.MemoryPoolInfoProcess(record, stack);

    record.eventRecord.type = RecordType::ADDR_INFO_RECORD;
    record.eventRecord.record.addrInfo = info;
    memoryStateRecord.MemoryAddrInfoProcess(record, stack);
}

TEST(MemoryStateRecordTest, memory_access_info_process_expect_success)
{
    auto record = Record{};
    record.eventRecord.type = RecordType::MEM_ACCESS_RECORD;
    auto memAccessRecord = MemAccessRecord{};
    
    memAccessRecord.eventType = AccessType::UNKNOWN;
    memAccessRecord.memType = OpType::ATEN;
    strncpy_s(memAccessRecord.name, sizeof(memAccessRecord.name),
              "ElewiseOperation", sizeof(memAccessRecord.name) - 1);
    strncpy_s(memAccessRecord.attr, sizeof(memAccessRecord.attr),
              "{dtype:FLOAT,format:ACL_ND,shape:1 2 }", sizeof(memAccessRecord.attr) - 1);
    memAccessRecord.timestamp = 7890;
    memAccessRecord.pid = 10;
    memAccessRecord.tid = 11;
    memAccessRecord.devId = 3;
    memAccessRecord.recordIndex = 1;
    memAccessRecord.addr = 1234;
    record.eventRecord.record.memAccessRecord = memAccessRecord;

    CallStackString stack{};
    Config config;
    MemoryStateRecord memoryStateRecord{config};
    auto key = std::make_pair("PTA", memAccessRecord.addr);
    memoryStateRecord.MemoryAccessInfoProcess(record, stack);
    ASSERT_EQ(memoryStateRecord.ptrMemoryInfoMap_.size(), 1);
    ASSERT_EQ(memoryStateRecord.ptrMemoryInfoMap_[key].size(), 1);

    memAccessRecord.eventType = AccessType::WRITE;
    memoryStateRecord.MemoryAccessInfoProcess(record, stack);
    ASSERT_EQ(memoryStateRecord.ptrMemoryInfoMap_.size(), 1);
    ASSERT_EQ(memoryStateRecord.ptrMemoryInfoMap_[key].size(), 2);

    memAccessRecord.eventType = AccessType::READ;
    memoryStateRecord.MemoryAccessInfoProcess(record, stack);
    ASSERT_EQ(memoryStateRecord.ptrMemoryInfoMap_.size(), 1);
    ASSERT_EQ(memoryStateRecord.ptrMemoryInfoMap_[key].size(), 3);
}

TEST(MemoryStateRecordTest, delete_memstateinfo_expect_success)
{
    Config config;
    MemoryStateRecord memoryStateRecord{config};
    auto key = std::make_pair("ATB", 1234);
    memoryStateRecord.ptrMemoryInfoMap_.insert({key, {}});
    memoryStateRecord.DeleteMemStateInfo(key);
    ASSERT_EQ(memoryStateRecord.ptrMemoryInfoMap_.size(), 0);
}

TEST(MemoryStateRecordTest, hal_memory_process_expect_success)
{
    Config config;
    MemoryStateRecord memoryStateRecord{config};
    auto record = MemOpRecord{};
    record.memType = MemOpType::FREE;
    uint64_t siz = 0;
    std::string type = "test";
    memoryStateRecord.HalMemProcess(record, siz, type);
}

TEST(MemoryStateRecordTest, get_halattr_hccl_expect_success)
{
    auto memRecordMalloc = MemOpRecord{};
    memRecordMalloc.devType = DeviceType::NPU;
    memRecordMalloc.modid = 3;
    memRecordMalloc.addr = 1234;
    memRecordMalloc.memType = MemOpType::MALLOC;
 
    Config config;
    BitField<decltype(config.analysisType)> analysisTypeBit;
    analysisTypeBit.setBit(static_cast<size_t>(AnalysisType::DECOMPOSE_ANALYSIS));
    MemoryStateRecord memoryStateRecord{config};
    MemRecordAttr attr = memoryStateRecord.GetMemInfoAttr(memRecordMalloc, 100);
}

TEST(MemoryStateRecordTest, pack_malloc_and_free_container_info)
{
    DumpContainer container;
    auto memPoolRecord = MemPoolRecord{};
    memPoolRecord.type = RecordType::TORCH_NPU_RECORD;
    memPoolRecord.memoryUsage.ptr = 123;
    memPoolRecord.recordIndex = 1;
    memPoolRecord.pid = 1234;
    memPoolRecord.tid = 1234;
    memPoolRecord.timeStamp = 1000;
    memPoolRecord.devId = 1;
    auto ptaMemUsage = MemoryUsage{};
    ptaMemUsage.ptr = 123;
    ptaMemUsage.allocSize = 100;
    ptaMemUsage.totalAllocated = 10000;
    ptaMemUsage.totalReserved = 30000;
    ptaMemUsage.totalActive = 10000;
    memPoolRecord.memoryUsage = ptaMemUsage;

    std::string memPoolType = "MALLOC";
    MemRecordAttr attr;

    Config config;
    BitField<decltype(config.analysisType)> analysisTypeBit;
    analysisTypeBit.setBit(static_cast<size_t>(AnalysisType::DECOMPOSE_ANALYSIS));
    MemoryStateRecord memoryStateRecord{config};
    memoryStateRecord.PackDumpContainer(container, memPoolRecord, memPoolType, attr);
}

TEST(MemoryStateRecordTest, pack_access_container_info)
{
    DumpContainer container;
    auto memAccessRecord = MemAccessRecord{};
    
    memAccessRecord.eventType = AccessType::UNKNOWN;
    memAccessRecord.memType = OpType::ATEN;
    strncpy_s(memAccessRecord.name, sizeof(memAccessRecord.name),
              "ElewiseOperation", sizeof(memAccessRecord.name) - 1);
    strncpy_s(memAccessRecord.attr, sizeof(memAccessRecord.attr),
              "{dtype:FLOAT,format:ACL_ND,shape:1 2 }", sizeof(memAccessRecord.attr) - 1);
    memAccessRecord.timestamp = 7890;
    memAccessRecord.pid = 10;
    memAccessRecord.tid = 11;
    memAccessRecord.devId = 3;
    memAccessRecord.recordIndex = 1;
    memAccessRecord.addr = 1234;

    std::string eventType = "WRITE";
    std::string attr = "test";

    Config config;
    BitField<decltype(config.analysisType)> analysisTypeBit;
    analysisTypeBit.setBit(static_cast<size_t>(AnalysisType::DECOMPOSE_ANALYSIS));
    MemoryStateRecord memoryStateRecord{config};
    memoryStateRecord.PackDumpContainer(container, memAccessRecord, eventType, attr);
}

TEST(MemoryStateRecordTest, update_leaks_defined_owner)
{
    std::string owner;
    std::string newOwner;
    Config config;
    MemoryStateRecord memoryStateRecord{config};

    owner = "PTA";
    newOwner = "@ops@aten";
    memoryStateRecord.UpdateLeaksDefinedOwner(owner, newOwner);

    owner = "PT";
    newOwner = "@ops@aten";
    memoryStateRecord.UpdateLeaksDefinedOwner(owner, newOwner);

    owner = "PTA@ops@aten";
    newOwner = "@model@weight";
    memoryStateRecord.UpdateLeaksDefinedOwner(owner, newOwner);
}
