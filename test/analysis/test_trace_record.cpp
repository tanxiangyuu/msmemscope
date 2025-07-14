// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#define private public
#include "trace_record.h"
#undef private
#include "record_info.h"
#include "config_info.h"
#include "securec.h"
#include "file.h"

#include <iostream>

using namespace Leaks;

bool ReadFile(const std::string &filePath, std::string &content)
{
    // 关闭文件，并将指针设为nullptr
    for (auto& pair : TraceRecord::GetInstance().traceFiles_) {
        if (pair.second.fp != nullptr) {
            fclose(pair.second.fp);
            pair.second.fp = nullptr;
        }
    }

    std::ifstream file(filePath);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        content += line + "\n";
    }

    file.close();
    return true;
}

bool RemoveDir(const std::string& dirPath)
{
    DIR* dir = opendir(dirPath.c_str());
    if (dir == nullptr) {
        return false;
    }

    // 清除路径下所有文件
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        std::string fullPath = dirPath + "/" + entry->d_name;

        struct stat statBuf;
        if (stat(fullPath.c_str(), &statBuf) != 0) {
            closedir(dir);
            return false;
        }

        if (S_ISDIR(statBuf.st_mode)) {
            if (!RemoveDir(fullPath)) {
                closedir(dir);
                return false;
            }
        } else {
            if (unlink(fullPath.c_str()) != 0) {
                closedir(dir);
                return false;
            }
        }
    }

    closedir(dir);

    // 删除空目录
    if (rmdir(dirPath.c_str()) != 0) {
        return false;
    }

    return true;
}

TEST(TraceRecord, process_mstx_mark_record)
{
    Utility::SetDirPath("leaksDumpResults", std::string(OUTPUT_PATH));
    TraceRecord::GetInstance().SetDirPath();
    auto buffer = RecordBuffer::CreateRecordBuffer<MstxRecord>(TLVBlockType::MARK_MESSAGE, "mark");
    MstxRecord* mstxRecord = buffer.Cast<MstxRecord>();
    mstxRecord->type = RecordType::MSTX_MARK_RECORD;
    mstxRecord->tid = 6;
    mstxRecord->pid = 8;
    mstxRecord->kernelIndex = 123;
    mstxRecord->rangeId = 0;
    mstxRecord->stepId = 0;
    mstxRecord->devId = 2;
    mstxRecord->markType = MarkType::MARK_A;

    TraceRecord::GetInstance().ProcessRecord(mstxRecord);
    std::string result = "{\n    \"ph\": \"i\",\n"
"    \"name\": \"mstx_mark\",\n"
"    \"pid\": 0,\n    \"tid\": 6,\n"
"    \"ts\": 123,\n    \"s\": \"p\",\n"
"    \"args\": {\n        \"message\": \"mark\"\n    }\n},\n";
    std::string fileContent;
    bool hasReadFile = ReadFile(
        TraceRecord::GetInstance().traceFiles_[Device{DeviceType::NPU, mstxRecord->devId}].filePath, fileContent);
    bool hasRemoveDir = RemoveDir(Utility::g_dirPath);
    EXPECT_NE(fileContent.find(result), std::string::npos);
    EXPECT_TRUE(hasReadFile && hasRemoveDir);
}

TEST(TraceRecord, process_mstx_record_with_report_host_memory)
{
    Utility::SetDirPath("leaksDumpResults", std::string(OUTPUT_PATH));
    TraceRecord::GetInstance().SetDirPath();
    auto buffer = RecordBuffer::CreateRecordBuffer<MstxRecord>(
        TLVBlockType::MARK_MESSAGE, "report host memory info start");
    MstxRecord* startMstxRecord = buffer.Cast<MstxRecord>();
    startMstxRecord->type = RecordType::MSTX_MARK_RECORD;
    startMstxRecord->tid = 6;
    startMstxRecord->pid = 8;
    startMstxRecord->kernelIndex = 123;
    startMstxRecord->rangeId = 1;
    startMstxRecord->stepId = 0;
    startMstxRecord->devId = 2;
    startMstxRecord->markType = MarkType::RANGE_START_A;

    auto buffer1 = RecordBuffer::CreateRecordBuffer<MstxRecord>();
    MstxRecord* endMstxRecord = buffer1.Cast<MstxRecord>();
    endMstxRecord->type = RecordType::MSTX_MARK_RECORD;
    endMstxRecord->tid = 6;
    endMstxRecord->pid = 8;
    endMstxRecord->kernelIndex = 133;
    endMstxRecord->rangeId = 1;
    endMstxRecord->stepId = 0;
    endMstxRecord->devId = 2;
    endMstxRecord->markType = MarkType::RANGE_END;

    TraceRecord::GetInstance().ProcessRecord(startMstxRecord);
    TraceRecord::GetInstance().ProcessRecord(endMstxRecord);
    std::string result = "{\n    \"ph\": \"i\",\n"
"    \"name\": \"mstx_range1_start\",\n"
"    \"pid\": 0,\n    \"tid\": 6,\n"
"    \"ts\": 123,\n    \"s\": \"p\",\n"
"    \"args\": {\n        \"message\": \"report host memory info start\"\n    }\n},\n{\n"
"    \"ph\": \"i\",\n"
"    \"name\": \"mstx_range1_end\",\n"
"    \"pid\": 0,\n    \"tid\": 6,\n"
"    \"ts\": 133,\n    \"s\": \"p\",\n"
"    \"args\": {\n        \"message\": \"N/A\"\n    }\n},\n";
    std::string fileContent;
    bool hasReadFile = ReadFile(
        TraceRecord::GetInstance().traceFiles_[Device{DeviceType::NPU, startMstxRecord->devId}].filePath, fileContent);
    bool hasRemoveDir = RemoveDir(Utility::g_dirPath);
    EXPECT_NE(fileContent.find(result), std::string::npos);
    EXPECT_TRUE(hasReadFile && hasRemoveDir);
}

TEST(TraceRecord, process_mstx_record_with_step_info)
{
    Utility::SetDirPath("leaksDumpResults", std::string(OUTPUT_PATH));
    TraceRecord::GetInstance().SetDirPath();
    auto buffer = RecordBuffer::CreateRecordBuffer<MstxRecord>(TLVBlockType::MARK_MESSAGE, "step start");
    MstxRecord* startMstxRecord = buffer.Cast<MstxRecord>();
    startMstxRecord->type = RecordType::MSTX_MARK_RECORD;
    startMstxRecord->tid = 6;
    startMstxRecord->pid = 8;
    startMstxRecord->kernelIndex = 123;
    startMstxRecord->rangeId = 2;
    startMstxRecord->stepId = 1;
    startMstxRecord->devId = 2;
    startMstxRecord->markType = MarkType::RANGE_START_A;

    auto buffer1 = RecordBuffer::CreateRecordBuffer<MstxRecord>();
    MstxRecord* endMstxRecord = buffer1.Cast<MstxRecord>();
    endMstxRecord->type = RecordType::MSTX_MARK_RECORD;
    endMstxRecord->tid = 6;
    endMstxRecord->pid = 8;
    endMstxRecord->kernelIndex = 133;
    endMstxRecord->rangeId = 2;
    endMstxRecord->stepId = 1;
    endMstxRecord->devId = 2;
    endMstxRecord->markType = MarkType::RANGE_END;

    TraceRecord::GetInstance().ProcessRecord(startMstxRecord);
    TraceRecord::GetInstance().ProcessRecord(endMstxRecord);
    std::string result = "{\n    \"ph\": \"i\",\n"
"    \"name\": \"mstx_step1_start\",\n"
"    \"pid\": 0,\n    \"tid\": 6,\n"
"    \"ts\": 123,\n    \"s\": \"p\",\n"
"    \"args\": {\n        \"message\": \"step start\"\n    }\n},\n{\n"
"    \"ph\": \"X\",\n"
"    \"name\": \"step 1\",\n"
"    \"pid\": 0,\n    \"tid\": 6,\n"
"    \"ts\": 123,\n    \"dur\": 10\n},\n{\n"
"    \"ph\": \"i\",\n"
"    \"name\": \"mstx_step1_end\",\n"
"    \"pid\": 0,\n    \"tid\": 6,\n"
"    \"ts\": 133,\n    \"s\": \"p\",\n"
"    \"args\": {\n        \"message\": \"N/A\"\n    }\n},\n";
    std::string fileContent;
    bool hasReadFile = ReadFile(
        TraceRecord::GetInstance().traceFiles_[Device{DeviceType::NPU, startMstxRecord->devId}].filePath, fileContent);
    bool hasRemoveDir = RemoveDir(Utility::g_dirPath);
    EXPECT_NE(fileContent.find(result), std::string::npos);
    EXPECT_TRUE(hasReadFile && hasRemoveDir);
}

TEST(TraceRecord, process_kernel_launch_record)
{
    Utility::SetDirPath("leaksDumpResults", std::string(OUTPUT_PATH));
    TraceRecord::GetInstance().SetDirPath();
    auto buffer = RecordBuffer::CreateRecordBuffer<KernelLaunchRecord>();
    KernelLaunchRecord* kernelLaunchRecord = buffer.Cast<KernelLaunchRecord>();
    kernelLaunchRecord->type = RecordType::KERNEL_LAUNCH_RECORD;
    kernelLaunchRecord->tid = 6;
    kernelLaunchRecord->pid = 8;
    kernelLaunchRecord->kernelLaunchIndex = 123;
    kernelLaunchRecord->devId = 2;

    std::string result = "{\n"
"    \"ph\": \"i\",\n"
"    \"name\": \"kernel_123\",\n"
"    \"pid\": 8,\n"
"    \"tid\": 6,\n"
"    \"ts\": 123,\n"
"    \"s\": \"p\"\n"
"},\n";

    TraceRecord::GetInstance().ProcessRecord(kernelLaunchRecord);
    std::string fileContent;
    bool hasReadFile = ReadFile(
        TraceRecord::GetInstance().traceFiles_[Device{DeviceType::NPU, kernelLaunchRecord->devId}].filePath,
        fileContent
    );
    bool hasRemoveDir = RemoveDir(Utility::g_dirPath);
    EXPECT_NE(fileContent.find(result), std::string::npos);
    EXPECT_TRUE(hasReadFile && hasRemoveDir);
}

TEST(TraceRecord, process_acl_itf_record)
{
    Utility::SetDirPath("leaksDumpResults", std::string(OUTPUT_PATH));
    TraceRecord::GetInstance().SetDirPath();
    auto buffer = RecordBuffer::CreateRecordBuffer<AclItfRecord>();
    AclItfRecord* aclItfRecord = buffer.Cast<AclItfRecord>();
    aclItfRecord->type = RecordType::ACL_ITF_RECORD;
    aclItfRecord->tid = 6;
    aclItfRecord->pid = 8;
    aclItfRecord->kernelIndex = 123;
    aclItfRecord->devId = 2;
    aclItfRecord->aclItfRecordIndex = 0;
    aclItfRecord->aclOpType = AclOpType::INIT;

    std::string result = "{\n"
"    \"ph\": \"i\",\n"
"    \"name\": \"acl_init\",\n"
"    \"pid\": 8,\n"
"    \"tid\": 6,\n"
"    \"ts\": 123,\n"
"    \"s\": \"p\"\n"
"},\n";

    TraceRecord::GetInstance().ProcessRecord(aclItfRecord);
    std::string fileContent;
    bool hasReadFile = ReadFile(
        TraceRecord::GetInstance().traceFiles_[Device{DeviceType::NPU, aclItfRecord->devId}].filePath, fileContent);
    bool hasRemoveDir = RemoveDir(Utility::g_dirPath);
    EXPECT_NE(fileContent.find(result), std::string::npos);
    EXPECT_TRUE(hasReadFile && hasRemoveDir);
}

TEST(TraceRecord, process_invalid_npu_memory_record)
{
    Utility::SetDirPath("leaksDumpResults", std::string(OUTPUT_PATH));
    TraceRecord::GetInstance().SetDirPath();
    auto buffer = RecordBuffer::CreateRecordBuffer<MemOpRecord>();
    MemOpRecord* freeMemOpRecord = buffer.Cast<MemOpRecord>();
    freeMemOpRecord->type = RecordType::MEMORY_RECORD;
    freeMemOpRecord->tid = 6;
    freeMemOpRecord->pid = 8;
    freeMemOpRecord->kernelIndex = 133;
    freeMemOpRecord->space = MemOpSpace::HOST;
    freeMemOpRecord->devType = DeviceType::NPU;
    freeMemOpRecord->devId = 2;
    freeMemOpRecord->subtype = RecordSubType::FREE;
    freeMemOpRecord->addr = 10000000;
    freeMemOpRecord->memSize = 0;

    TraceRecord::GetInstance().ProcessRecord(freeMemOpRecord);
    std::string result = "";
    std::string fileContent;
    bool hasReadFile = ReadFile(
        TraceRecord::GetInstance().traceFiles_[Device{DeviceType::NPU, freeMemOpRecord->devId}].filePath, fileContent);
    bool hasRemoveDir = RemoveDir(TraceRecord::GetInstance().dirPath_);
    EXPECT_EQ(result, fileContent);
    EXPECT_FALSE(hasReadFile);
}

TEST(TraceRecord, process_invalid_cpu_memory_record)
{
    Utility::SetDirPath("leaksDumpResults", std::string(OUTPUT_PATH));
    TraceRecord::GetInstance().SetDirPath();
    auto buffer = RecordBuffer::CreateRecordBuffer<MemOpRecord>();
    MemOpRecord* freeMemOpRecord = buffer.Cast<MemOpRecord>();
    freeMemOpRecord->type = RecordType::MEMORY_RECORD;
    freeMemOpRecord->tid = 6;
    freeMemOpRecord->pid = 8;
    freeMemOpRecord->kernelIndex = 133;
    freeMemOpRecord->space = MemOpSpace::HOST;
    freeMemOpRecord->devType = DeviceType::CPU;
    freeMemOpRecord->devId = 0;
    freeMemOpRecord->subtype = RecordSubType::FREE;
    freeMemOpRecord->addr = 10000000;
    freeMemOpRecord->memSize = 0;

    TraceRecord::GetInstance().ProcessRecord(freeMemOpRecord);
    std::string result = "";
    std::string fileContent;
    bool hasReadFile = ReadFile(
        TraceRecord::GetInstance().traceFiles_[Device{DeviceType::CPU, freeMemOpRecord->devId}].filePath, fileContent);
    bool hasRemoveDir = RemoveDir(TraceRecord::GetInstance().dirPath_);
    EXPECT_EQ(result, fileContent);
    EXPECT_FALSE(hasReadFile);
}

TEST(TraceRecord, process_cpu_memory_record)
{
    Utility::SetDirPath("leaksDumpResults", std::string(OUTPUT_PATH));
    TraceRecord::GetInstance().SetDirPath();
    auto buffer = RecordBuffer::CreateRecordBuffer<MemOpRecord>();
    MemOpRecord* mallocMemOpRecord = buffer.Cast<MemOpRecord>();
    mallocMemOpRecord->type = RecordType::MEMORY_RECORD;
    mallocMemOpRecord->tid = 6;
    mallocMemOpRecord->pid = 8;
    mallocMemOpRecord->kernelIndex = 123;
    mallocMemOpRecord->space = MemOpSpace::HOST;
    mallocMemOpRecord->devType = DeviceType::CPU;
    mallocMemOpRecord->devId = 0;
    mallocMemOpRecord->subtype = RecordSubType::MALLOC;
    mallocMemOpRecord->addr = 10000000;
    mallocMemOpRecord->memSize = 10000;


    auto buffer1 = RecordBuffer::CreateRecordBuffer<MemOpRecord>();
    MemOpRecord* freeMemOpRecord = buffer1.Cast<MemOpRecord>();
    freeMemOpRecord->type = RecordType::MEMORY_RECORD;
    freeMemOpRecord->tid = 6;
    freeMemOpRecord->pid = 8;
    freeMemOpRecord->kernelIndex = 133;
    freeMemOpRecord->space = MemOpSpace::INVALID;
    freeMemOpRecord->devType = DeviceType::CPU;
    freeMemOpRecord->devId = 0;
    freeMemOpRecord->subtype = RecordSubType::FREE;
    freeMemOpRecord->addr = 10000000;
    freeMemOpRecord->memSize = 0;

    TraceRecord::GetInstance().ProcessRecord(mallocMemOpRecord);
    TraceRecord::GetInstance().ProcessRecord(freeMemOpRecord);
    std::string result = "{\n"
"    \"ph\": \"C\",\n    \"name\": \"memory\",\n"
"    \"pid\": 8,\n    \"tid\": 6,\n    \"ts\": 123,\n"
"    \"args\": {\n        \"size\": 10000\n    }\n},\n{\n"
"    \"ph\": \"C\",\n    \"name\": \"memory\",\n"
"    \"pid\": 8,\n    \"tid\": 6,\n    \"ts\": 133,\n"
"    \"args\": {\n        \"size\": 0\n    }\n},\n";
    std::string fileContent;
    bool hasReadFile = ReadFile(
        TraceRecord::GetInstance().traceFiles_[Device{DeviceType::CPU, mallocMemOpRecord->devId}].filePath,
        fileContent);
    bool hasRemoveDir = RemoveDir(Utility::g_dirPath);
    EXPECT_NE(fileContent.find(result), std::string::npos);
    EXPECT_TRUE(hasReadFile && hasRemoveDir);
}

TEST(TraceRecord, process_hal_host_memory_record)
{
    Utility::SetDirPath("leaksDumpResults", std::string(OUTPUT_PATH));
    TraceRecord::GetInstance().SetDirPath();
    auto buffer = RecordBuffer::CreateRecordBuffer<MemOpRecord>();
    MemOpRecord* mallocMemOpRecord = buffer.Cast<MemOpRecord>();
    mallocMemOpRecord->type = RecordType::MEMORY_RECORD;
    mallocMemOpRecord->tid = 6;
    mallocMemOpRecord->pid = 8;
    mallocMemOpRecord->kernelIndex = 123;
    mallocMemOpRecord->space = MemOpSpace::HOST;
    mallocMemOpRecord->devType = DeviceType::NPU;
    mallocMemOpRecord->devId = 2;
    mallocMemOpRecord->subtype = RecordSubType::MALLOC;
    mallocMemOpRecord->addr = 10000000;
    mallocMemOpRecord->memSize = 10000;

    auto buffer1 = RecordBuffer::CreateRecordBuffer<MemOpRecord>();
    MemOpRecord* freeMemOpRecord = buffer1.Cast<MemOpRecord>();
    freeMemOpRecord->type = RecordType::MEMORY_RECORD;
    freeMemOpRecord->tid = 6;
    freeMemOpRecord->pid = 8;
    freeMemOpRecord->kernelIndex = 133;
    freeMemOpRecord->space = MemOpSpace::INVALID;
    freeMemOpRecord->devType = DeviceType::NPU;
    freeMemOpRecord->devId = 2;
    freeMemOpRecord->subtype = RecordSubType::FREE;
    freeMemOpRecord->addr = 10000000;
    freeMemOpRecord->memSize = 0;

    TraceRecord::GetInstance().ProcessRecord(mallocMemOpRecord);
    TraceRecord::GetInstance().ProcessRecord(freeMemOpRecord);
    std::string result = "{\n"
"    \"ph\": \"C\",\n    \"name\": \"pin memory\",\n"
"    \"pid\": 8,\n    \"tid\": 6,\n    \"ts\": 123,\n"
"    \"args\": {\n        \"size\": 10000\n    }\n},\n{\n"
"    \"ph\": \"C\",\n    \"name\": \"pin memory\",\n"
"    \"pid\": 8,\n    \"tid\": 6,\n    \"ts\": 133,\n"
"    \"args\": {\n        \"size\": 0\n    }\n},\n";

    std::string fileContent;
    bool hasReadFile = ReadFile(
        TraceRecord::GetInstance().traceFiles_[Device{DeviceType::CPU, 0}].filePath,
        fileContent);
    bool hasRemoveDir = RemoveDir(Utility::g_dirPath);
    EXPECT_NE(fileContent.find(result), std::string::npos);
    EXPECT_TRUE(hasReadFile && hasRemoveDir);
}

TEST(TraceRecord, process_torch_memory_record)
{
    Utility::SetDirPath("leaksDumpResults", std::string(OUTPUT_PATH));
    TraceRecord::GetInstance().SetDirPath();
    auto buffer = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* record = buffer.Cast<MemPoolRecord>();
    record->type = RecordType::TORCH_NPU_RECORD;
    record->tid = 6;
    record->pid = 8;
    record->kernelIndex = 123;
    MemoryUsage memoryUsage = MemoryUsage{};
    memoryUsage.totalAllocated = 10;
    memoryUsage.totalReserved = 30;
    record->memoryUsage = memoryUsage;
    record->devId = 2;

    std::string result = "{\n"
"    \"ph\": \"C\",\n"
"    \"name\": \"torch reserved memory\",\n"
"    \"pid\": 8,\n"
"    \"tid\": 6,\n"
"    \"ts\": 123,\n"
"    \"args\": {\n        \"size\": 30\n    }\n},\n{\n"
"    \"ph\": \"C\",\n"
"    \"name\": \"torch allocated memory\",\n"
"    \"pid\": 8,\n"
"    \"tid\": 6,\n"
"    \"ts\": 123,\n"
"    \"args\": {\n        \"size\": 10\n    }\n},\n";

    TraceRecord::GetInstance().ProcessRecord(record);
    std::string fileContent;
    bool hasReadFile = ReadFile(
        TraceRecord::GetInstance().traceFiles_[Device{DeviceType::NPU, record->devId}].filePath, fileContent);
    bool hasRemoveDir = RemoveDir(Utility::g_dirPath);
    EXPECT_NE(fileContent.find(result), std::string::npos);
    EXPECT_TRUE(hasReadFile && hasRemoveDir);
}

TEST(TraceRecord, process_torch_mem_leak_info)
{
    Utility::SetDirPath("leaksDumpResults", std::string(OUTPUT_PATH));
    TraceRecord::GetInstance().SetDirPath();
    auto info = TorchMemLeakInfo{};
    info.devId = 2;
    info.kernelIndex = 123;
    info.duration = 10;
    info.addr = 10000000;
    info.size = 1000;

    std::string result = "{\n"
"    \"ph\": \"X\",\n"
"    \"name\": \"mem 10000000 leak\",\n"
"    \"pid\": 1,\n"
"    \"tid\": 0,\n"
"    \"ts\": 123,\n"
"    \"dur\": 10,\n"
"    \"args\": {\n"
"        \"addr\": 10000000,\"size\": 1000\n"
"    }\n"
"},\n";

    TraceRecord::GetInstance().ProcessTorchMemLeakInfo(info);
    std::string fileContent;
    bool hasReadFile = ReadFile(
        TraceRecord::GetInstance().traceFiles_[Device{DeviceType::NPU, info.devId}].filePath, fileContent);
    bool hasRemoveDir = RemoveDir(Utility::g_dirPath);
    EXPECT_NE(fileContent.find(result), std::string::npos);
    EXPECT_TRUE(hasReadFile && hasRemoveDir);
}

TEST(TraceRecord, set_metadata_event)
{
    Utility::SetDirPath("leaksDumpResults", std::string(OUTPUT_PATH));
    TraceRecord::GetInstance().SetDirPath();
    Device device = Device{DeviceType::NPU, 0};
    TraceRecord::GetInstance().truePids_[device] = {1234};

    std::vector<std::string> results(5);
    results[0] = "{\n    \"ph\": \"M\",\n    \"name\": \"process_sort_index\",\n"
"    \"pid\": 1234,\n    \"tid\": 0,\n"
"    \"args\": {\n        \"sort_index\": 0\n    }\n},\n";
    results[1] = "{\n    \"ph\": \"M\",\n    \"name\": \"process_name\",\n"
"    \"pid\": 0,\n    \"tid\": 0,\n"
"    \"args\": {\n        \"name\": \"mstx\"\n    }\n},\n";
    results[2] = "{\n    \"ph\": \"M\",\n    \"name\": \"process_sort_index\",\n"
"    \"pid\": 0,\n    \"tid\": 0,\n"
"    \"args\": {\n        \"sort_index\": 1\n    }\n},\n";
    results[3] = "{\n    \"ph\": \"M\",\n    \"name\": \"process_name\",\n"
"    \"pid\": 1,\n    \"tid\": 0,\n"
"    \"args\": {\n        \"name\": \"leak\"\n    }\n},\n";
    results[4] = "{\n    \"ph\": \"M\",\n    \"name\": \"process_sort_index\",\n"
"    \"pid\": 1,\n    \"tid\": 0,\n"
"    \"args\": {\n        \"sort_index\": 2\n    }\n},\n";

    TraceRecord::GetInstance().SetMetadataEvent(device);
    std::string fileContent;
    bool hasReadFile = ReadFile(TraceRecord::GetInstance().traceFiles_[device].filePath, fileContent);
    bool hasRemoveDir = RemoveDir(Utility::g_dirPath);
    for (auto result : results) {
        EXPECT_NE(fileContent.find(result), std::string::npos);
    }
    EXPECT_TRUE(hasReadFile && hasRemoveDir);
}

TEST(TraceRecord, set_dir_path)
{
    Utility::SetDirPath("MyPath", std::string(OUTPUT_PATH));
    TraceRecord::GetInstance().SetDirPath();
    EXPECT_EQ(TraceRecord::GetInstance().dirPath_, "MyPath/" + std::string(TRACE_FILE));
}

TEST(TraceRecord, process_mindspore_memory_record)
{
    Utility::SetDirPath("leaksDumpResults", std::string(OUTPUT_PATH));
    TraceRecord::GetInstance().SetDirPath();
    auto buffer = RecordBuffer::CreateRecordBuffer<MemPoolRecord>();
    MemPoolRecord* record = buffer.Cast<MemPoolRecord>();
    record->type = RecordType::MINDSPORE_NPU_RECORD;
    record->tid = 6;
    record->pid = 8;
    record->kernelIndex = 123;
    MemoryUsage memoryUsage = MemoryUsage{};
    memoryUsage.totalAllocated = 10;
    memoryUsage.totalReserved = 30;
    record->memoryUsage = memoryUsage;
    record->devId = 2;

    std::string result = "{\n"
"    \"ph\": \"C\",\n"
"    \"name\": \"mindspore reserved memory\",\n"
"    \"pid\": 8,\n"
"    \"tid\": 6,\n"
"    \"ts\": 123,\n"
"    \"args\": {\n        \"size\": 30\n    }\n},\n{\n"
"    \"ph\": \"C\",\n"
"    \"name\": \"mindspore allocated memory\",\n"
"    \"pid\": 8,\n"
"    \"tid\": 6,\n"
"    \"ts\": 123,\n"
"    \"args\": {\n        \"size\": 10\n    }\n},\n";

    TraceRecord::GetInstance().ProcessRecord(record);
    std::string fileContent;
    bool hasReadFile = ReadFile(
        TraceRecord::GetInstance().traceFiles_[Device{DeviceType::NPU, record->devId}].filePath, fileContent);
    bool hasRemoveDir = RemoveDir(Utility::g_dirPath);
    EXPECT_NE(fileContent.find(result), std::string::npos);
    EXPECT_TRUE(hasReadFile && hasRemoveDir);
}

TEST(TraceRecord, safe_write_string_failed)
{
    std::string str = "";
    Device device1 = Device{DeviceType::NPU, GD_INVALID_NUM};
    TraceRecord::GetInstance().SafeWriteString(str, device1);

    Device device2 = Device{DeviceType::NPU, -1};
    TraceRecord::GetInstance().SafeWriteString(str, device2);
}