// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include <sstream>
#include "file.h"
#include "utils.h"
#include "data_handler.h"
namespace Leaks {

DataHandler::DataHandler(const Config config)
{
    config_ = config;
}

CsvHandler::CsvHandler(const Config config, DumpClass dumpType) : DataHandler(config), dumpType_(dumpType)
{
    InitSetParm();
}

void CsvHandler::InitSetParm()
{
    switch (dumpType_) {
        case DumpClass::LEAKS_RECORD: {
            prefix_ = "leaks_dump_";
            dirPath_ = std::string(config_.outputDir) + "/" + std::string(DUMP_FILE);
            std::string cStack = config_.enableCStack ? ",Call Stack(C)" : "";
            std::string pyStack = config_.enablePyStack ? ",Call Stack(Python)" : "";
            csvHeader_ = LEAKS_HEADERS + pyStack + cStack + "\n";
            break;
        }
        case DumpClass::PYTHON_TRACE:
            prefix_ = "python_trace_" + std::to_string(Utility::GetPid()) + "_";
            dirPath_ = std::string(config_.outputDir) + "/" + std::string(DUMP_FILE);
            csvHeader_ = TRACE_HEADERS;
            break;
        default:
            LOG_ERROR("Unsupported data type : %d\n", static_cast<int>(dumpType_));
            break;
    }
}
bool CsvHandler::Init()
{
    return Utility::CreateCsvFile(&file_, dirPath_, prefix_, csvHeader_);
}

bool CsvHandler::Write(DumpDataClass *data, const CallStackString &stack)
{
    if (!data) {
        LOG_ERROR("Null data pointer");
        return false;
    }

    switch (data->dumpType) {
        case DumpClass::LEAKS_RECORD:
            return WriteDumpRecord(static_cast<DumpContainer*>(data), stack);
        case DumpClass::PYTHON_TRACE:
            return WriteTraceEvent(static_cast<TraceEvent*>(data));
        default:
            LOG_ERROR("Unsupported data type : %d\n", static_cast<int>(data->dumpType));
            return false;
    }
    return false;
}

bool CsvHandler::WriteDumpRecord(const DumpContainer* container, const CallStackString& stack)
{
    std::string pid = container->pid == INVALID_PROCESSID ? "N/A" : std::to_string(container->pid);
    std::string tid = container->tid == INVALID_THREADID ? "N/A" : std::to_string(container->tid);
    if (!Utility::Fprintf(file_, "%lu,%s,%s,%s,%lu,%s,%s,%s,%s,%s",
        container->id, container->event.c_str(), container->eventType.c_str(), container->name.c_str(),
        container->timestamp, pid.c_str(), tid.c_str(), container->deviceId.c_str(),
        container->addr.c_str(), container->attr.c_str())) {
        return false;
    }
    if (config_.enablePyStack && !Utility::Fprintf(file_, ",%s", stack.pyStack.c_str())) {
        return false;
    }
    if (config_.enableCStack && !Utility::Fprintf(file_, ",%s", stack.cStack.c_str())) {
        return false;
    }
    if (!Utility::Fprintf(file_, "\n")) {
        return false;
    }
    return true;
}

bool CsvHandler::WriteTraceEvent(const TraceEvent* event)
{
    std::string startTime = event->startTs ? std::to_string(event->startTs) : "N/A";
    std::string endTime = event->endTs ? std::to_string(event->endTs) : "N/A";
    if (!Utility::Fprintf(file_, "%s,%s,%s,%lu,%lu\n", event->info.c_str(), startTime.c_str(),
        endTime.c_str(), event->tid, event->pid)) {
        return false;
    }
    return true;
}

bool CsvHandler::Read(std::vector<DumpContainer>& data)
{
    // 还未适配
    return false;
}

CsvHandler::~CsvHandler()
{
    if (file_ != nullptr) {
        std::fclose(file_);
        file_ = nullptr;
    }
}

DbHandler::DbHandler(const Config config, DumpClass dumpType) : DataHandler(config), dumpType_(dumpType)
{
    InitSetParm();
}

void DbHandler::InitSetParm()
{
    switch (dumpType_) {
        case DumpClass::LEAKS_RECORD: {
            std::vector<std::pair<std::string, std::string>> schema = DUMP_RECORD_TABLE_SQL;
            leakColumns_ = ParserHeader(DUMP_RECORD_TABLE_SQL);
            if (config_.enablePyStack) {
                schema.emplace_back("Call Stack(Python)", "TEXT");
                leakColumns_.push_back("Call Stack(Python)");
            }
            if (config_.enableCStack) {
                schema.emplace_back("Call Stack(C)", "TEXT");
                leakColumns_.push_back("Call Stack(C)");
            }
            tableName_ = DUMP_RECORD_TABLE;
            dbHeader_ = BuildCreateStatement(tableName_, schema);
            if (!Init()) {
                LOG_ERROR("Sqlite create error: %s", Sqlite3Errmsg(dataFileDb_));
                break;
            }
            std::string insertSql = BuildInsertStatement(DUMP_RECORD_TABLE, leakColumns_);
            int resultCount1 = Sqlite3PrepareV2(dataFileDb_, insertSql.c_str(), -1, &insertLeakStmt_, nullptr);
            if (resultCount1 != SQLITE_OK) {
                LOG_ERROR("Sqlite prepare error: %s", Sqlite3Errmsg(dataFileDb_));
                insertLeakStmt_ = nullptr;
            }
            break;
        }
        case DumpClass::PYTHON_TRACE: {
            tableName_ = "python_trace_" + std::to_string(Utility::GetPid());
            dbHeader_ = BuildCreateStatement(tableName_, PYTHON_TRACE_TABLE_SQL);
            traceColumns_ = ParserHeader(PYTHON_TRACE_TABLE_SQL);
            if (!Init()) {
                LOG_ERROR("Sqlite create error: %s", Sqlite3Errmsg(dataFileDb_));
                break;
            }
            std::string insertSql = BuildInsertStatement(tableName_, traceColumns_);
            int resultCount2 = Sqlite3PrepareV2(dataFileDb_, insertSql.c_str(), -1, &insertTraceStmt_, nullptr);
            if (resultCount2 != SQLITE_OK) {
                LOG_ERROR("Sqlite prepare error: %s", Sqlite3Errmsg(dataFileDb_));
                insertTraceStmt_ = nullptr;
            }
            break;
        }
        default:
            LOG_ERROR("Unsupported data type : %d\n", static_cast<int>(dumpType_));
            break;
    }
}

bool DbHandler::Init()
{
    std::string filePath = std::string(config_.outputDir) + "/" + Leaks::DUMP_FILE + "/" + config_.dbFileName;
    return Utility::CreateDbFile(&dataFileDb_, filePath, tableName_, dbHeader_);
}

bool DbHandler::Write(DumpDataClass *data, const CallStackString &stack)
{
    if (!data) {
        LOG_ERROR("Null data pointer");
        return false;
    }
    switch (data->dumpType) {
        case DumpClass::LEAKS_RECORD:
            return WriteDumpRecord(static_cast<DumpContainer*>(data), stack);
        case DumpClass::PYTHON_TRACE:
            return WriteTraceEvent(static_cast<TraceEvent*>(data), tableName_);
        default:
            LOG_ERROR("Unsupported data type : %d\n", static_cast<int>(data->dumpType));
            return false;
    }
    return false;
}

bool DbHandler::WriteDumpRecord(const DumpContainer* container, const CallStackString& stack)
{
    std::string attrJson = FixJson(container->attr);
    int paramIndex = 1;
    Sqlite3BindInt64(insertLeakStmt_, paramIndex++, container->id);
    Sqlite3BindText(insertLeakStmt_, paramIndex++, container->event.c_str(), -1, SQLITE_STATIC);
    Sqlite3BindText(insertLeakStmt_, paramIndex++, container->eventType.c_str(), -1, SQLITE_STATIC);
    Sqlite3BindText(insertLeakStmt_, paramIndex++, container->name.c_str(), -1, SQLITE_STATIC);
    Sqlite3BindInt64(insertLeakStmt_, paramIndex++, container->timestamp);
    Sqlite3BindInt(insertLeakStmt_, paramIndex++, container->pid);
    Sqlite3BindInt(insertLeakStmt_, paramIndex++, container->tid);
    Sqlite3BindText(insertLeakStmt_, paramIndex++, container->deviceId.c_str(), -1, SQLITE_STATIC);
    Sqlite3BindText(insertLeakStmt_, paramIndex++, container->addr.c_str(), -1, SQLITE_STATIC);
    Sqlite3BindText(insertLeakStmt_, paramIndex++, attrJson.c_str(), -1, SQLITE_STATIC);
    if (config_.enablePyStack) {
        Sqlite3BindText(insertLeakStmt_, paramIndex++, stack.pyStack.c_str(), -1, SQLITE_STATIC);
    }
    if (config_.enableCStack) {
        Sqlite3BindText(insertLeakStmt_, paramIndex++, stack.cStack.c_str(), -1, SQLITE_STATIC);
    }
    Sqlite3BusyTimeout(dataFileDb_, SQLITE_TIME_OUT);
    int rc = Sqlite3Step(insertLeakStmt_);
    if (rc != SQLITE_DONE) {
        LOG_ERROR("Sqlite insert error in leaks dump: %s  %d", Sqlite3Errmsg(dataFileDb_), getpid());
        Sqlite3Reset(insertLeakStmt_);
        return false;
    }
    Sqlite3Reset(insertLeakStmt_);
    return true;
}

bool DbHandler::WriteTraceEvent(const TraceEvent* event, const std::string &tableName)
{
    std::string startTime = event->startTs ? std::to_string(event->startTs) : "N/A";
    std::string endTime = event->endTs ? std::to_string(event->endTs) : "N/A";
    int paramIndex = 1;
    Sqlite3BindText(insertTraceStmt_, paramIndex++, event->info.c_str(), -1, SQLITE_STATIC);
    Sqlite3BindText(insertTraceStmt_, paramIndex++, startTime.c_str(), -1, SQLITE_STATIC);
    Sqlite3BindText(insertTraceStmt_, paramIndex++, endTime.c_str(), -1, SQLITE_STATIC);
    Sqlite3BindInt64(insertTraceStmt_, paramIndex++, event->tid);
    Sqlite3BindInt64(insertTraceStmt_, paramIndex++, event->pid);

    Sqlite3BusyTimeout(dataFileDb_, SQLITE_TIME_OUT);
    int rc = Sqlite3Step(insertTraceStmt_);
    if (rc != SQLITE_DONE) {
        LOG_ERROR("Sqlite insert error in python trace: %s", Sqlite3Errmsg(dataFileDb_));
        Sqlite3Reset(insertTraceStmt_);
        return false;
    }
    Sqlite3Reset(insertTraceStmt_);
    return true;
}

bool DbHandler::Read(std::vector<DumpContainer>& data)
{
    // 还未适配
    return true;
}

DbHandler::~DbHandler()
{
    if (dataFileDb_ != nullptr) {
        Sqlite3Exec(dataFileDb_, "PRAGMA wal_checkpoint(FULL);", nullptr, nullptr, nullptr);
        // 提交任何未完成的事务
        Sqlite3Exec(dataFileDb_, "COMMIT;", nullptr, nullptr, nullptr);
        if (insertLeakStmt_ != nullptr) {
            Sqlite3Finalize(insertLeakStmt_);
        }
        if (insertTraceStmt_ != nullptr) {
            Sqlite3Finalize(insertTraceStmt_);
        }
        int rc = Sqlite3Close(dataFileDb_);
        if (rc != SQLITE_OK) {
            LOG_ERROR("Sqlite close error: %s", Sqlite3Errmsg(dataFileDb_));
        }
        dataFileDb_ = nullptr;
    }
}

std::string BuildInsertStatement(const std::string& table, const std::vector<std::string>& columns)
{
    std::ostringstream oss;
    oss << "INSERT INTO " << table << " (";
    for (size_t i = 0; i < columns.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "\"" << columns[i] << "\"";
    }
    oss << ") VALUES (";
    for (size_t i = 0; i < columns.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "?";
    }
    oss << ");";
    return oss.str();
}

std::string BuildCreateStatement(const std::string& table,
    const std::vector<std::pair<std::string, std::string>>& columns)
{
    std::ostringstream oss;
    oss << "CREATE TABLE IF NOT EXISTS " << table << " (";

    for (size_t i = 0; i < columns.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << "\"" << columns[i].first << "\" " << columns[i].second;
    }

    oss << ");";
    return oss.str();
}

std::unique_ptr<DataHandler> MakeDataHandler(Config config, DumpClass data)
{
    switch (config.dataFormat) {
        case static_cast<uint8_t>(DataFormat::CSV):
            return std::unique_ptr<DataHandler>(new CsvHandler(config, data));
            break;
        case static_cast<uint8_t>(DataFormat::DB):
            return std::unique_ptr<DataHandler>(new DbHandler(config, data));
            break;
        default:
            LOG_ERROR("Unsupported format: %lu", config.dataFormat);
            return nullptr;
    }
}

// 将"\"{addr:20616937226752,size:28160,...}\"" 转成标准JSON字符串{"addr":"20616937226752","size":"28160",...}
std::string FixJson(const std::string& input)
{
    std::string json = input;
    uint32_t subPlace = 2;
    if (json.size() >= subPlace && json.front() == '"' && json.back() == '"') {
        json = json.substr(1, json.length() - subPlace);
    }
    size_t pos = 0;
    while ((pos = json.find("\\\"", pos)) != std::string::npos) {
        json.replace(pos, subPlace, "\"");
    }
    if (json.size() >= subPlace && json.front() == '{' && json.back() == '}') {
        json = json.substr(1, json.length() - subPlace);
    }
    std::istringstream iss(json);
    std::string token;
    std::vector<std::string> parts;

    while (std::getline(iss, token, ',')) {
        parts.push_back(token);
    }
    std::ostringstream oss;
    oss << "{";
    for (size_t i = 0; i < parts.size(); ++i) {
        std::istringstream partStream(parts[i]);
        std::string key;
        std::string value;

        if (std::getline(partStream, key, ':') && std::getline(partStream, value)) {
            oss << "\"" << key << "\":\"" << value << "\"";
            if (i != parts.size() - 1) {
                oss << ",";
            }
        }
    }
    oss << "}";
    return oss.str();
}

std::vector<std::string> ParserHeader(const std::vector<std::pair<std::string, std::string>>& header)
{
    std::vector<std::string> result;
    result.reserve(header.size());
    for (const auto& item : header) {
        result.push_back(item.first);
    }
    return result;
}
};
