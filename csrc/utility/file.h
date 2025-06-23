// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef LEAKS_UTILITY_FILE_H
#define LEAKS_UTILITY_FILE_H

#include <unistd.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <linux/limits.h>
#include "path.h"
#include "config_info.h"
#include "umask_guard.h"
#include "ustring.h"
#include "sqlite_loader.h"

namespace Utility {
    constexpr uint32_t DIRMOD = 0750;
    constexpr uint32_t DEFAULT_UMASK_FOR_CSV_FILE = 0177;
    constexpr uint32_t DEFAULT_UMASK_FOR_DB_FILE = 0177;
    constexpr uint32_t DEFAULT_UMASK_FOR_BIN_FILE = 0177;
    extern std::string g_dirPath;
    constexpr uint64_t MAX_INPUT_FILE_SIZE = 1UL << 33; // 8GB

    inline void SetDirPath(const std::string& dirPath, const std::string& defaultDirPath)
    {
        if (dirPath.length() > PATH_MAX) {
            std::cout << "[msleaks] Error: Path " << dirPath << " length exceeds the maximum length:"
                      << PATH_MAX << "." << std::endl;
            return;
        }
        if (dirPath.empty()) {
            Utility::Path path = Utility::Path{defaultDirPath};
            Utility::Path realPath = path.Resolved();
            if (realPath.ErrorOccured()) { return; }
            g_dirPath = realPath.ToString();
        } else {
            g_dirPath = dirPath;
        }
    }

    inline bool MakeDir(const std::string& dirPath)
    {
        if (dirPath.empty()) {
            std::cout << "[msleaks] Error: The directory path is empty." << std::endl;
            return false;
        }

        if (access(dirPath.c_str(), F_OK) != -1) {
            return true;
        }
        
        size_t pos = dirPath[0] != '/' ? 0 : 1;
        std::string tempPath = dirPath;
        
        // 遍历路径中的每一部分，递归创建父目录
        while ((pos = tempPath.find('/', pos)) != std::string::npos) {
            std::string partPath = tempPath.substr(0, pos);
            pos++;
            if (access(partPath.c_str(), F_OK) != -1) {
                continue;
            }
            if (mkdir(partPath.c_str(), DIRMOD) != 0) {
                std::cout << "[msleaks] Error: Cannot create dir " << partPath << " ." << std::endl;
                return false;
            }
        }

        if (mkdir(dirPath.c_str(), DIRMOD) != 0) {
            std::cout << "[msleaks] Error: Cannot create dir " << dirPath << " ." << std::endl;
            return false;
        }
        return true;
    }

    inline bool Exist(const std::string &path)
    {
        if (path.empty()) {
            std::cout << "[msleaks] Error: The file path is empty." << std::endl;
            return false;
        }
        return access(path.c_str(), F_OK) == 0;
    }

    // 多线程情况下调用，需加锁保护
    bool CreateCsvFile(FILE **filefp, std::string dirPath, std::string fileName, std::string headers);
    bool CreateDbFile(sqlite3 **filefp, std::string filePath, std::string tableName, std::string tableCreateSql);

    template <typename... Args>
    inline bool Fprintf(FILE* fp, const std::string &format, const Args& ...args)
    {
        if (fp == nullptr) {
            std::cout << "[msleaks] Error: Fail to write data to file, fp is NULL" << std::endl;
            return false;
        }
        if (int fpRes = fprintf(fp, format.c_str(), args...) < 0) {
            std::cout << "[msleaks] Error: Fail to write data to file, errno: " << fpRes << std::endl;
            return false;
        }
        return true;
    }

    // 输入+输出文件专属校验：文件大小
    inline bool IsFileSizeSafe(const std::string& path)
    {
        struct stat buffer;
        if (lstat(path.c_str(), &buffer) != 0) {
            std::cout << "[msleaks] Error: Error getting file state for " << path << "." << std::endl;
            return false;
        }

        if (!S_ISREG(buffer.st_mode)) {
            std::cout << "[msleaks] Error: File " << path << " is not a regular file." << std::endl;
            return false;
        }

        if (buffer.st_size > static_cast<int64_t>(MAX_INPUT_FILE_SIZE)) {
            std::cout << "[msleaks] Error: File " << path << " exceeds maximum size ("
                      << MAX_INPUT_FILE_SIZE << " bytes)." << std::endl;
            return false;
        }
        return true;
    }

    /// 输出文件校验
    bool CheckFileBeforeCreate(const std::string &path);
    bool FileExists(const std::string& filePath);
    bool TableExists(sqlite3 *filefp, std::string tableName);
    bool CreateDbPath(Leaks::Config &config, const std::string &fileName);
    bool CreateDbTable(sqlite3 *filefp, std::string tableCreateSql);
    /// 创建文件
    FILE* CreateFileWithUmask(const std::string &path, const std::string &mode, mode_t mask);
    FILE* CreateFile(const std::string &dir, const std::string &name, mode_t mask);

}

#endif