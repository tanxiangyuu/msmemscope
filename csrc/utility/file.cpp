// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include <dlfcn.h>
#include <securec.h>
#include "utils.h"
#include "file.h"

namespace Utility {

    std::string g_dirPath;

    // Log创建之前不要使用LOG_***
    bool CheckFileBeforeCreate(const std::string &path)
    {
        Path curPath = Path(path).Absolute();
        std::string curPathStr = curPath.ToString();
        std::string users;

        if (!curPath.Exists()) {
            std::cout << "[msleaks] Error: The path " << curPathStr << " do not exist." << std::endl;
            return false;
        }
        if (CheckStrIsStartsWithInvalidChar(curPathStr.c_str())) {
            std::cout << "[msleaks] Error: The path " << curPathStr << " is invalid." << std::endl;
            return false;
        }
        if (!curPath.IsReadable()) {
            std::cout << "[msleaks] Error: The file path " << curPathStr << " is not readable." << std::endl;
            return false;
        }
        if (!curPath.IsValidLength()) {
            std::cout << "[msleaks] Error: The length of file path " << curPathStr
                      << " exceeds the maximum length." << std::endl;
            return false;
        }
        if (!curPath.IsValidDepth()) {
            std::cout << "[msleaks] Error: The depth of file path " << curPathStr
                      << " exceeds the maximum depth." << std::endl;
            return false;
        }
        if (curPath.IsSoftLink()) {
            std::cout << "[msleaks] Error: The file path " << curPathStr
                      << " is invalid: soft link is not allowed." << std::endl;
            return false;
        }
        if (!curPath.IsPermissionValid()) {
            std::cout << "[msleaks] Error: The file path " << curPathStr
                      << " is invalid: permission is not valid." << std::endl;
            return false;
        }
        return true;
    }

    FILE* CreateFileWithUmask(const std::string &path, const std::string &mode, mode_t mask)
    {
        if (path.empty()) {
            return nullptr;
        }
        UmaskGuard guard{mask};
        FILE *fp = fopen(path.c_str(), mode.c_str());
        return fp;
    }

    FILE* CreateFile(const std::string &dir, const std::string &name, mode_t mask)
    {
        if (!MakeDir(dir)) {
            return nullptr;
        }
        std::string filePath = dir + "/" + name;
        if (!CheckFileBeforeCreate(dir)) {
            return nullptr;
        }
        return CreateFileWithUmask(filePath, "a", mask);
    }

    bool CreateCsvFile(FILE **filefp, std::string dirPath, std::string fileName, std::string headers)
    {
        if (*filefp == nullptr) {
            fileName = fileName + Utility::GetDateStr() + ".csv";
            std::string filePath = dirPath + "/" + fileName;
            FILE* fp = CreateFile(dirPath, fileName, DEFAULT_UMASK_FOR_CSV_FILE);
            if (fp != nullptr) {
                std::cout << "[msleaks] Info: create file " << filePath << "." << std::endl;
                fprintf(fp, "%s", headers.c_str());
                *filefp = fp;
            } else {
                std::cout << "[msleaks] Error: open file " << filePath << " failed." << std::endl;
                return false;
            }
        }
        return true;
    }

    bool CreateDbPath(Leaks::Config &config, const std::string &fileName)
    {
        std::string name = fileName + "_" + Utility::GetDateStr() + ".db";
        std::string path = std::string(config.outputDir) + "/" + Leaks::DUMP_FILE;
        if (!MakeDir(path)) {
            return false;
        }
        if (!CheckFileBeforeCreate(path)) {
            return false;
        }
        std::string filePath = std::string(config.outputDir) + "/" + Leaks::DUMP_FILE + "/" + name;
        if (strncpy_s(config.dbFileDir, sizeof(config.dbFileDir),
            filePath.c_str(), sizeof(config.dbFileDir) - 1) != EOK) {
            std::cout << "[msleaks] strncpy_s FAILED DB" << std::endl;
        }
        config.dbFileDir[sizeof(config.dbFileDir) - 1] = '\0';
        if (std::string(config.dbFileDir).length() > PATH_MAX) {
            std::cout << "[msleaks] Error: Path " << std::string(config.dbFileDir)
                        << " length exceeds the maximum length:" << PATH_MAX << "." << std::endl;
            return false;
        }
        return true;
    }

    bool TableExists(sqlite3 *filefp, std::string tableName)
    {
        std::string sql = "SELECT name FROM sqlite_master WHERE type='table' AND name=?";
        sqlite3_stmt* stmt;
        int rc = Sqlite3PrepareV2(filefp, sql.c_str(), -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            return false;
        }
        rc = Sqlite3BindText(stmt, 1, tableName.c_str(), -1, SQLITE_STATIC);
        if (rc != SQLITE_OK) {
            Sqlite3Finalize(stmt);
            return false;
        }
        Sqlite3BusyTimeout(filefp, Leaks::SQLITE_TIME_OUT);
        rc = Sqlite3Step(stmt);
        bool exists = (rc == SQLITE_ROW);
        Sqlite3Finalize(stmt);
        return exists;
    }
    
    bool CreateDbTable(sqlite3 *filefp, std::string tableCreateSql)
    {
        Sqlite3BusyTimeout(filefp, Leaks::SQLITE_TIME_OUT);
        int rc = Sqlite3Exec(filefp, tableCreateSql.c_str(), nullptr, nullptr, nullptr);
        if (rc != SQLITE_OK) {
            std::cout << "[msleaks] Create SQLTable error: " << Sqlite3Errmsg(filefp) << std::endl;
            return false;
        }
        return true;
    }

    bool CreateDbFile(sqlite3 **filefp, std::string filePath, std::string tableName, std::string tableCreateSql)
    {
        if (*filefp == nullptr) {
            sqlite3* db = nullptr;
            int rc = Sqlite3Open(filePath.c_str(), &db);
            if (rc != SQLITE_OK) {
                std::cout << "[msleaks] Cannot open database: " << Sqlite3Errmsg(db) << std::endl;
                if (db != nullptr) {
                    Sqlite3Close(db);
                }
                return false;
            }
            if (CreateDbTable(db, tableCreateSql)) {
                std::cout << "[msleaks] Info: create dbfile " << filePath << "." << std::endl;
                *filefp = db;
            } else {
                Sqlite3Close(db);
                return false;
            }
        } else if (!TableExists(*filefp, tableName) && !CreateDbTable(*filefp, tableCreateSql)) {
            return false;
        }
        return true;
    }
}  // namespace Utility