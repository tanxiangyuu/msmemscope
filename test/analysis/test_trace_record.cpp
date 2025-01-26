// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <dirent.h>
#define private public
#include "trace_record.h"
#undef private
#include "record_info.h"
#include "config_info.h"

#include <iostream>

using namespace Leaks;

static std::string g_traceDirPath = "leaksDumpResults";

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
        if (entry->d_type == DT_DIR) {
            return false;
        }
        if (remove(fullPath.c_str()) != 0) {
            return false;
        }
    }
    closedir(dir);

    // 删除空目录
    if (rmdir(dirPath.c_str()) != 0) {
        return false;
    }

    return true;
}

TEST(TraceRecord, process_mstx_record)
{
    auto startRecord = EventRecord{};
    startRecord.type = RecordType::MSTX_MARK_RECORD;
    auto startMstxRecord = MstxRecord{};
    startMstxRecord.tid = 0;
    startMstxRecord.pid = 0;
    startMstxRecord.timeStamp = 123;
    startMstxRecord.rangeId = 0;
    startMstxRecord.devId = 0;
    startMstxRecord.markType = MarkType::RANGE_START_A;
    startRecord.record.mstxRecord = startMstxRecord;

    auto endRecord = EventRecord{};
    endRecord.type = RecordType::MSTX_MARK_RECORD;
    auto endMstxRecord = MstxRecord{};
    endMstxRecord.tid = 0;
    endMstxRecord.pid = 0;
    endMstxRecord.timeStamp = 133;
    endMstxRecord.rangeId = 0;
    endMstxRecord.devId = 0;
    endMstxRecord.markType = MarkType::RANGE_END;
    endMstxRecord.markMessage[0] = 'e';
    endRecord.record.mstxRecord = endMstxRecord;

    TraceRecord::GetInstance().ProcessRecord(startRecord);
    TraceRecord::GetInstance().ProcessRecord(endRecord);
    std::string result = "{\n    \"ph\": \"i\",\n"
"    \"name\": \"mstx_range_start_0\",\n"
"    \"pid\": 0,\n    \"tid\": 0,\n"
"    \"ts\": 123,\n    \"s\": \"p\"\n},\n{\n"
"    \"ph\": \"X\",\n"
"    \"name\": \"step 0\",\n"
"    \"pid\": 0,\n    \"tid\": 0,\n"
"    \"ts\": 123,\n    \"dur\": 10\n},\n{\n"
"    \"ph\": \"i\",\n"
"    \"name\": \"mstx_range_end_0\",\n"
"    \"pid\": 0,\n    \"tid\": 0,\n"
"    \"ts\": 133,\n    \"s\": \"p\",\n"
"    \"args\": {\n        \"message\": \"e\"\n    }\n},\n";
    std::string fileContent;
    bool hasReadFile = ReadFile(TraceRecord::GetInstance().traceFiles_[startMstxRecord.devId].filePath, fileContent);
    bool hasRemoveDir = RemoveDir("./" + g_traceDirPath);
    EXPECT_NE(fileContent.find(result), std::string::npos);
    EXPECT_TRUE(hasReadFile && hasRemoveDir);
}

TEST(TraceRecord, process_kernel_launch_record)
{
    auto record = EventRecord{};
    record.type = RecordType::KERNEL_LAUNCH_RECORD;
    auto kernelLaunchRecord = KernelLaunchRecord{};
    kernelLaunchRecord.tid = 0;
    kernelLaunchRecord.pid = 0;
    kernelLaunchRecord.timeStamp = 123;
    kernelLaunchRecord.devId = 0;
    kernelLaunchRecord.kernelLaunchIndex = 0;
    record.record.kernelLaunchRecord = kernelLaunchRecord;

    std::string result = "{\n"
"    \"ph\": \"i\",\n"
"    \"name\": \"kernel_0\",\n"
"    \"pid\": 0,\n"
"    \"tid\": 0,\n"
"    \"ts\": 123,\n"
"    \"s\": \"p\"\n"
"},\n";

    TraceRecord::GetInstance().ProcessRecord(record);
    std::string fileContent;
    bool hasReadFile = ReadFile(
        TraceRecord::GetInstance().traceFiles_[kernelLaunchRecord.devId].filePath,
        fileContent
    );
    bool hasRemoveDir = RemoveDir("./" + g_traceDirPath);
    EXPECT_NE(fileContent.find(result), std::string::npos);
    EXPECT_TRUE(hasReadFile && hasRemoveDir);
}

TEST(TraceRecord, process_acl_itf_record)
{
    auto record = EventRecord{};
    record.type = RecordType::ACL_ITF_RECORD;
    auto aclItfRecord = AclItfRecord{};
    aclItfRecord.tid = 0;
    aclItfRecord.pid = 0;
    aclItfRecord.timeStamp = 123;
    aclItfRecord.devId = 0;
    aclItfRecord.aclItfRecordIndex = 0;
    record.record.aclItfRecord = aclItfRecord;

    std::string result = "{\n"
"    \"ph\": \"i\",\n"
"    \"name\": \"acl_0\",\n"
"    \"pid\": 0,\n"
"    \"tid\": 0,\n"
"    \"ts\": 123,\n"
"    \"s\": \"p\"\n"
"},\n";

    TraceRecord::GetInstance().ProcessRecord(record);
    std::string fileContent;
    bool hasReadFile = ReadFile(TraceRecord::GetInstance().traceFiles_[aclItfRecord.devId].filePath, fileContent);
    bool hasRemoveDir = RemoveDir("./" + g_traceDirPath);
    EXPECT_NE(fileContent.find(result), std::string::npos);
    EXPECT_TRUE(hasReadFile && hasRemoveDir);
}

TEST(TraceRecord, process_host_memory_record)
{
    auto mallocRecord = EventRecord{};
    mallocRecord.type = RecordType::MEMORY_RECORD;
    auto mallocMemOpRecord = MemOpRecord{};
    mallocMemOpRecord.tid = 0;
    mallocMemOpRecord.pid = 0;
    mallocMemOpRecord.timeStamp = 123;
    mallocMemOpRecord.space = MemOpSpace::HOST;
    mallocMemOpRecord.devId = 0;
    mallocMemOpRecord.memType = MemOpType::MALLOC;
    mallocMemOpRecord.addr = 10000000;
    mallocMemOpRecord.memSize = 10000;
    mallocRecord.record.memoryRecord = mallocMemOpRecord;

    auto freeRecord = EventRecord{};
    freeRecord.type = RecordType::MEMORY_RECORD;
    auto freeMemOpRecord = MemOpRecord{};
    freeMemOpRecord.tid = 0;
    freeMemOpRecord.pid = 0;
    freeMemOpRecord.timeStamp = 133;
    freeMemOpRecord.space = MemOpSpace::HOST;
    freeMemOpRecord.devId = 0;
    freeMemOpRecord.memType = MemOpType::FREE;
    freeMemOpRecord.addr = 10000000;
    freeMemOpRecord.memSize = 10000;
    freeRecord.record.memoryRecord = freeMemOpRecord;

    TraceRecord::GetInstance().ProcessRecord(mallocRecord);
    TraceRecord::GetInstance().ProcessRecord(freeRecord);
    std::string result = "";
    std::string fileContent;
    bool hasReadFile = ReadFile(TraceRecord::GetInstance().traceFiles_[mallocMemOpRecord.devId].filePath, fileContent);
    bool hasRemoveDir = RemoveDir("./" + g_traceDirPath);
    EXPECT_EQ(result, fileContent);
    EXPECT_FALSE(hasReadFile);
}

TEST(TraceRecord, process_device_memory_record)
{
    auto mallocRecord = EventRecord{};
    mallocRecord.type = RecordType::MEMORY_RECORD;
    auto mallocMemOpRecord = MemOpRecord{};
    mallocMemOpRecord.tid = 0;
    mallocMemOpRecord.pid = 0;
    mallocMemOpRecord.timeStamp = 123;
    mallocMemOpRecord.space = MemOpSpace::DEVICE;
    mallocMemOpRecord.devId = 0;
    mallocMemOpRecord.memType = MemOpType::MALLOC;
    mallocMemOpRecord.addr = 10000000;
    mallocMemOpRecord.memSize = 10000;
    mallocRecord.record.memoryRecord = mallocMemOpRecord;

    auto freeRecord = EventRecord{};
    freeRecord.type = RecordType::MEMORY_RECORD;
    auto freeMemOpRecord = MemOpRecord{};
    freeMemOpRecord.tid = 0;
    freeMemOpRecord.pid = 0;
    freeMemOpRecord.timeStamp = 133;
    freeMemOpRecord.space = MemOpSpace::INVALID;
    freeMemOpRecord.devId = 0;
    freeMemOpRecord.memType = MemOpType::FREE;
    freeMemOpRecord.addr = 10000000;
    freeMemOpRecord.memSize = 10000;
    freeRecord.record.memoryRecord = freeMemOpRecord;

    std::string result = "{\n"
"    \"ph\": \"C\",\n"
"    \"name\": \"device memory\",\n"
"    \"pid\": 0,\n"
"    \"tid\": 0,\n"
"    \"ts\": 123,\n"
"    \"args\": {\n        \"size\": 10000\n    }\n},\n{\n"
"    \"ph\": \"C\",\n"
"    \"name\": \"device memory\",\n"
"    \"pid\": 0,\n"
"    \"tid\": 0,\n"
"    \"ts\": 133,\n"
"    \"args\": {\n        \"size\": 0\n    }\n},\n";

    TraceRecord::GetInstance().ProcessRecord(mallocRecord);
    TraceRecord::GetInstance().ProcessRecord(freeRecord);
    std::string fileContent;
    bool hasReadFile = ReadFile(TraceRecord::GetInstance().traceFiles_[mallocMemOpRecord.devId].filePath, fileContent);
    bool hasRemoveDir = RemoveDir("./" + g_traceDirPath);
    EXPECT_NE(fileContent.find(result), std::string::npos);
    EXPECT_TRUE(hasReadFile && hasRemoveDir);
}

TEST(TraceRecord, process_torch_memory_record)
{
    auto record = EventRecord{};
    record.type = RecordType::TORCH_NPU_RECORD;
    auto torchNpuRecord = TorchNpuRecord{};
    torchNpuRecord.tid = 0;
    torchNpuRecord.pid = 0;
    torchNpuRecord.timeStamp = 123;
    MemoryUsage memoryUsage = MemoryUsage{};
    memoryUsage.totalAllocated = 10;
    memoryUsage.totalReserved = 30;
    memoryUsage.totalActive = 20;
    memoryUsage.deviceIndex = 0;
    torchNpuRecord.memoryUsage = memoryUsage;
    record.record.torchNpuRecord = torchNpuRecord;

    std::string result = "{\n"
"    \"ph\": \"C\",\n"
"    \"name\": \"operators reserved\",\n"
"    \"pid\": 0,\n"
"    \"tid\": 0,\n"
"    \"ts\": 123,\n"
"    \"args\": {\n        \"size\": 30\n    }\n},\n{\n"
"    \"ph\": \"C\",\n"
"    \"name\": \"operators active\",\n"
"    \"pid\": 0,\n"
"    \"tid\": 0,\n"
"    \"ts\": 123,\n"
"    \"args\": {\n        \"size\": 20\n    }\n},\n{\n"
"    \"ph\": \"C\",\n"
"    \"name\": \"operators allocated\",\n"
"    \"pid\": 0,\n"
"    \"tid\": 0,\n"
"    \"ts\": 123,\n"
"    \"args\": {\n        \"size\": 10\n    }\n},\n";

    TraceRecord::GetInstance().ProcessRecord(record);
    std::string fileContent;
    bool hasReadFile = ReadFile(TraceRecord::GetInstance().traceFiles_[memoryUsage.deviceIndex].filePath, fileContent);
    bool hasRemoveDir = RemoveDir("./" + g_traceDirPath);
    EXPECT_NE(fileContent.find(result), std::string::npos);
    EXPECT_TRUE(hasReadFile && hasRemoveDir);
}

TEST(TraceRecord, process_torch_mem_leak_info)
{
    auto info = TorchMemLeakInfo{};
    info.devId = 0;
    info.timestamp = 123;
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
    bool hasReadFile = ReadFile(TraceRecord::GetInstance().traceFiles_[info.devId].filePath, fileContent);
    bool hasRemoveDir = RemoveDir("./" + g_traceDirPath);
    EXPECT_NE(fileContent.find(result), std::string::npos);
    EXPECT_TRUE(hasReadFile && hasRemoveDir);
}

TEST(TraceRecord, set_metadata_event)
{
    int32_t devId = 0;
    TraceRecord::GetInstance().truePids_[devId] = {1234};

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

    TraceRecord::GetInstance().SetMetadataEvent(devId);
    std::string fileContent;
    bool hasReadFile = ReadFile(TraceRecord::GetInstance().traceFiles_[devId].filePath, fileContent);
    bool hasRemoveDir = RemoveDir("./" + g_traceDirPath);
    for (auto result : results) {
        EXPECT_NE(fileContent.find(result), std::string::npos);
    }
    EXPECT_TRUE(hasReadFile && hasRemoveDir);
}