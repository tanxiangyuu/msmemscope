// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef FILE_HANDLER_H
#define FILE_HANDLER_H

#include <string>
#include <vector>
#include <memory>
#include <cstdio>
#include <sqlite3.h>
#include <mutex>
#include <unordered_map>
#include "config_info.h"
#include "record_info.h"
#include "utils.h"
#include "file.h"
#include "utility/log.h"

namespace Leaks {

const std::vector<std::pair<std::string, std::string>> DUMP_RECORD_TABLE_SQL = {
    {"ID", "INTEGER"},
    {"Event", "TEXT"},
    {"Event Type", "TEXT"},
    {"Name", "TEXT"},
    {"Timestamp(ns)", "INTEGER"},
    {"Process Id", "INTEGER"},
    {"Thread Id", "INTEGER"},
    {"Device Id", "TEXT"},
    {"Ptr", "TEXT"},
    {"Attr", "TEXT"}
};

const std::vector<std::pair<std::string, std::string>> PYTHON_TRACE_TABLE_SQL = {
    {"FuncInfo", "TEXT"},
    {"StartTime(ns)", "TEXT"},
    {"EndTime(ns)", "TEXT"},
    {"Thread Id", "INTEGER"},
    {"Process Id", "INTEGER"}
};

const std::string DUMP_RECORD_TABLE = "leaks_dump";
const std::string PYTHON_TRACE_TABLE = "python_trace";

class DumpDataClass {
public:
    virtual ~DumpDataClass() = default;
    explicit DumpDataClass(DumpClass type) : dumpType(type) {}
    DumpClass dumpType;
};

// dump数据结构
struct DumpContainer : public DumpDataClass {
    DumpContainer() : DumpDataClass(DumpClass::LEAKS_RECORD) {}
    uint64_t id;
    std::string event;
    std::string eventType;
    std::string name;
    uint64_t timeStamp;
    uint64_t pid;
    uint64_t tid;
    std::string deviceId;
    std::string addr;
    std::string owner = "";
    std::string callStack = "";
    std::string attr = "";
};

struct TraceEvent : public DumpDataClass {
    TraceEvent() : DumpDataClass(DumpClass::PYTHON_TRACE) {}
    TraceEvent(const TraceEvent&) = default;
    TraceEvent& operator=(const TraceEvent&) = default;
    TraceEvent(
        uint64_t startTs,
        uint64_t endTs,
        uint64_t tid,
        uint64_t pid,
        const std::string& info,
        const std::string& hash
    ) : DumpDataClass(DumpClass::PYTHON_TRACE), startTs(startTs), endTs(endTs),
        tid(tid), pid(pid), info(info), hash(hash) {}

    uint64_t startTs = 0;
    uint64_t endTs = 0;
    uint64_t tid;
    uint64_t pid;
    std::string info;
    std::string hash;
};

// DumpHandler类主要用于将analyzer分析的数据dump至csv或者db文件
class DataHandler {
public:
    virtual ~DataHandler() = default;
    virtual bool Init() = 0;
    virtual bool Write(DumpDataClass *data, const CallStackString &stack) = 0;
    virtual bool Read(std::vector<DumpContainer>& data) = 0;

protected:
    explicit DataHandler(const Config config);
    Config config_;

private:
    DataHandler(const DataHandler&) = delete;
    DataHandler& operator=(const DataHandler&) = delete;
    DataHandler(DataHandler&& other) = delete;
    DataHandler& operator=(DataHandler&& other) = delete;
};

class CsvHandler : public DataHandler {
public:
    ~CsvHandler() override;
    explicit CsvHandler(const Config config, DumpClass data);
    bool Init() override;
    bool Write(DumpDataClass *data, const CallStackString &stack) override;
    bool Read(std::vector<DumpContainer>& data) override;

private:
    void InitSetParm();
    bool WriteDumpRecord(const DumpContainer* container, const CallStackString& stack);
    bool WriteTraceEvent(const TraceEvent* event);
    FILE *file_ = nullptr;
    std::string csvHeader_;
    std::string prefix_;
    DumpClass dumpType_;
    std::string dirPath_;
};

class DbHandler : public DataHandler {
public:
    explicit DbHandler(const Config config, DumpClass data);
    ~DbHandler() override;
    bool Init() override;
    bool Write(DumpDataClass *data, const CallStackString &stack) override;
    bool Read(std::vector<DumpContainer>& data) override;

private:
    void InitSetParm();
    bool WriteDumpRecord(const DumpContainer* container, const CallStackString& stack);
    bool WriteTraceEvent(const TraceEvent* event, const std::string &tableName);
    sqlite3 *dataFileDb_ = nullptr;
    std::string dbHeader_;
    std::string tableName_;
    std::string dirPath_;
    DumpClass dumpType_;
};

std::string BuildInsertStatement(const std::string& table, const std::vector<std::string>& columns);
std::string BuildCreateStatement(const std::string& table,
    const std::vector<std::pair<std::string, std::string>>& columns);

std::unique_ptr<DataHandler> MakeDataHandler(Config config, DumpClass data);
std::string FixJson(const std::string& input);
}
#endif