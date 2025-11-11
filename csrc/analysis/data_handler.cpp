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

// csv handler
CsvHandler::CsvHandler(const Config config, DataType dataType, std::string devId) : DataHandler(config),
    dataType_(dataType), devId_(devId)
{
    InitSetParm();
}

void CsvHandler::InitSetParm()
{
    switch (dataType_) {
        case DataType::LEAKS_EVENT: {
            prefix_ = CSV_FILE_PREFIX;
            csvHeader_ = LEAKS_HEADERS;
            break;
        }
        case DataType::PYTHON_TRACE_EVENT:
            prefix_ = PYTHON_TRACE_FILE_PREFIX;
            csvHeader_ = TRACE_HEADERS;
            break;
        default:
            LOG_ERROR("Unsupported data type : %d\n", static_cast<int>(dataType_));
            break;
    }
}

bool CsvHandler::Init()
{
    std::lock_guard<std::mutex> lock(csvFileMutex_);
    return Utility::FileCreateManager::GetInstance(config_.outputDir).CreateCsvFile(&file_, devId_, prefix_,
        DUMP_DIR, csvHeader_);
}

bool CsvHandler::Write(std::shared_ptr<DataBase> data)
{
    if (!data) {
        LOG_ERROR("Null data pointer");
        return false;
    }

    if (!Init()) {
        LOG_ERROR("Create csv file failed.");
        return false;
    }

    switch (data->GetDataType()) {
        case DataType::LEAKS_EVENT: {
            auto event = std::dynamic_pointer_cast<EventBase>(data);
            if (event) {
                return WriteDumpRecord(event);
            }
            break;
        }
        case DataType::PYTHON_TRACE_EVENT: {
            auto event = std::dynamic_pointer_cast<TraceEvent>(data);
            if (event) {
                return WriteTraceEvent(event);
            }
            break;
        }
        default:
            LOG_ERROR("Unsupported data type : %d\n", static_cast<int>(data->GetDataType()));
            return false;
    }
    return false;
}

bool CsvHandler::WriteDumpRecord(std::shared_ptr<EventBase>& event)
{
    std::lock_guard<std::mutex> lock(dumpFileMutex_);
    std::string pid = event->pid == INVALID_PROCESSID ? "N/A" : std::to_string(event->pid);
    std::string tid = event->tid == INVALID_THREADID ? "N/A" : std::to_string(event->tid);
    std::string eventType = EVENT_BASE_TYPE_MAP.find(event->eventType) == EVENT_BASE_TYPE_MAP.end()
        ? "N/A" : EVENT_BASE_TYPE_MAP.at(event->eventType);
    std::string eventSubType = EVENT_SUB_TYPE_MAP.find(event->eventSubType) == EVENT_SUB_TYPE_MAP.end()
        ? "N/A" : EVENT_SUB_TYPE_MAP.at(event->eventSubType);
    std::string addr = (event->eventType == EventBaseType::MALLOC
        || event->eventType == EventBaseType::FREE
        || event->eventType == EventBaseType::ACCESS) ? Uint64ToHexString(event->addr) : "N/A";
    if (!Utility::Fprintf(file_, "%lu,%s,%s,%s,%lu,%s,%s,%s,%s,%s,%s,%s",
        event->id, eventType.c_str(), eventSubType.c_str(), event->name.c_str(), event->timestamp,
        pid.c_str(), tid.c_str(), event->device.c_str(), addr.c_str(), event->attr.c_str(),
        event->pyCallStack.c_str(), event->cCallStack.c_str())) {
        return false;
    }

    if (!Utility::Fprintf(file_, "\n")) {
        return false;
    }
    return true;
}

bool CsvHandler::WriteTraceEvent(std::shared_ptr<TraceEvent>& event)
{
    std::lock_guard<std::mutex> lock(traceFileMutex_);
    std::string startTime = event->startTs ? std::to_string(event->startTs) : "N/A";
    std::string endTime = event->endTs ? std::to_string(event->endTs) : "N/A";
    if (!Utility::Fprintf(file_, "%s,%s,%s,%lu,%lu\n", event->info.c_str(), startTime.c_str(),
        endTime.c_str(), event->tid, event->pid)) {
        return false;
    }
    return true;
}

CsvHandler::~CsvHandler()
{
    if (file_ != nullptr) {
        std::fclose(file_);
        file_ = nullptr;
    }
}

DbHandler::DbHandler(const Config config, DataType dataType, std::string devId) : DataHandler(config),
    dataType_(dataType), devId_(devId)
{
    InitSetParm();
}

void DbHandler::InitSetParm()
{
    switch (dataType_) {
        case DataType::LEAKS_EVENT: {
            std::vector<std::pair<std::string, std::string>> schema = DUMP_RECORD_TABLE_SQL;
            leakColumns_ = ParserHeader(DUMP_RECORD_TABLE_SQL);

            schema.emplace_back("Call Stack(Python)", "TEXT");
            leakColumns_.push_back("Call Stack(Python)");
            schema.emplace_back("Call Stack(C)", "TEXT");
            leakColumns_.push_back("Call Stack(C)");

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
        case DataType::PYTHON_TRACE_EVENT: {
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
            LOG_ERROR("Unsupported data type : %d\n", static_cast<int>(dataType_));
            break;
    }
}

bool DbHandler::Init()
{
    std::lock_guard<std::mutex> lock(dbFileMutex_);
    return Utility::FileCreateManager::GetInstance(config_.outputDir).CreateDbFile(&dataFileDb_, devId_,
        CSV_FILE_PREFIX, DUMP_DIR, tableName_, dbHeader_);
}

bool DbHandler::Write(std::shared_ptr<DataBase> data)
{
    if (!data) {
        LOG_ERROR("Null data pointer");
        return false;
    }

    if (!Init()) {
        LOG_ERROR("Create db file failed.");
        return false;
    }

    switch (data->GetDataType()) {
        case DataType::LEAKS_EVENT: {
            auto event = std::dynamic_pointer_cast<EventBase>(data);
            if (event) {
                return WriteDumpRecord(event);
            }
            break;
        }
        case DataType::PYTHON_TRACE_EVENT: {
            auto event = std::dynamic_pointer_cast<TraceEvent>(data);
            if (event) {
                return WriteTraceEvent(event, tableName_);
            }
            break;
        }
        default:
            LOG_ERROR("Unsupported data type : %d\n", static_cast<int>(data->GetDataType()));
            return false;
    }
    return false;
}

bool DbHandler::WriteDumpRecord(std::shared_ptr<EventBase>& event)
{
    std::string eventType = EVENT_BASE_TYPE_MAP.find(event->eventType) == EVENT_BASE_TYPE_MAP.end()
        ? "N/A" : EVENT_BASE_TYPE_MAP.at(event->eventType);
    std::string eventSubType = EVENT_SUB_TYPE_MAP.find(event->eventSubType) == EVENT_SUB_TYPE_MAP.end()
        ? "N/A" : EVENT_SUB_TYPE_MAP.at(event->eventSubType);
    std::string addr = (event->eventType == EventBaseType::MALLOC
        || event->eventType == EventBaseType::FREE
        || event->eventType == EventBaseType::ACCESS) ? Uint64ToHexString(event->addr) : "N/A";
    std::string attrJson = FixJson(event->attr);
    int paramIndex = 1;
    std::lock_guard<std::mutex> lock(dbFileMutex_);
    if (!insertLeakStmt_) {
        LOG_ERROR("Sqlite prepare failed.");
        return false;
    }
    Sqlite3BindInt64(insertLeakStmt_, paramIndex++, event->id);
    Sqlite3BindText(insertLeakStmt_, paramIndex++, eventType.c_str(), -1, SQLITE_STATIC);
    Sqlite3BindText(insertLeakStmt_, paramIndex++, eventSubType.c_str(), -1, SQLITE_STATIC);
    Sqlite3BindText(insertLeakStmt_, paramIndex++, event->name.c_str(), -1, SQLITE_STATIC);
    Sqlite3BindInt64(insertLeakStmt_, paramIndex++, event->timestamp);
    Sqlite3BindInt(insertLeakStmt_, paramIndex++, event->pid);
    Sqlite3BindInt(insertLeakStmt_, paramIndex++, event->tid);
    Sqlite3BindText(insertLeakStmt_, paramIndex++, event->device.c_str(), -1, SQLITE_STATIC);
    Sqlite3BindText(insertLeakStmt_, paramIndex++, addr.c_str(), -1, SQLITE_STATIC);
    Sqlite3BindText(insertLeakStmt_, paramIndex++, attrJson.c_str(), -1, SQLITE_STATIC);
    Sqlite3BindText(insertLeakStmt_, paramIndex++, event->pyCallStack.c_str(), -1, SQLITE_STATIC);
    Sqlite3BindText(insertLeakStmt_, paramIndex++, event->cCallStack.c_str(), -1, SQLITE_STATIC);
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

bool DbHandler::WriteTraceEvent(std::shared_ptr<TraceEvent>& event, const std::string &tableName)
{
    std::lock_guard<std::mutex> lock(dbFileMutex_);
    if (!insertTraceStmt_) {
        LOG_ERROR("Sqlite prepare failed.");
        return false;
    }
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

std::unique_ptr<DataHandler> MakeDataHandler(Config config, DataType data, std::string devId)
{
    switch (config.dataFormat) {
        case static_cast<uint8_t>(DataFormat::CSV):
            return std::unique_ptr<DataHandler>(new CsvHandler(config, data, devId));
            break;
        case static_cast<uint8_t>(DataFormat::DB):
            return std::unique_ptr<DataHandler>(new DbHandler(config, data, devId));
            break;
        default:
            LOG_ERROR("Unsupported format: %lu", config.dataFormat);
            return nullptr;
    }
}

// 将"\"{addr:20616937226752,size:28160,...}\"" 转成标准JSON字符串{"addr":"20616937226752","size":"28160",...}
std::string FixJson(const std::string& input)
{
    std::string str = input;
    uint32_t minSize = 4;
    if (str.size() >= minSize) {
        str = str.substr(2, str.length() - minSize);
    }
    size_t pos = 0;
    std::vector<std::string> parts;

    while (pos < str.length()) {
        size_t colonPos = str.find(':', pos);
        if (colonPos == std::string::npos) {
            parts.push_back(str.substr(pos, str.length() - pos));
            break;
        }
        size_t lastCommaPos = str.rfind(',', colonPos);
        if (lastCommaPos == std::string::npos || lastCommaPos < pos) {
            // 没有找到逗号或者逗号在pos之前
            parts.push_back(str.substr(pos, colonPos - pos));
        } else {
            parts.push_back(str.substr(pos, lastCommaPos - pos));
            parts.push_back(str.substr(lastCommaPos + 1, colonPos - lastCommaPos - 1));
        }
        pos = colonPos + 1;
    }

    std::ostringstream oss;
    auto partsNum = parts.size();
    oss << "{";
    for (size_t i = 0; i < partsNum; i++) {
        oss << "\"" << parts[i] << "\":";
        i++;
        if (i >= partsNum) {
            oss << "\"\"";
            break;
        }
        oss << "\"" << parts[i] << "\"";
        if (i != partsNum - 1) {
            oss << ",";
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

std::string Uint64ToHexString(uint64_t value)
{
    std::stringstream ss;
    ss << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
    return ss.str();
}
};
