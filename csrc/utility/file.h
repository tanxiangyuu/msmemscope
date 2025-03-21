// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef LEAKS_UTILITY_FILE_H
#define LEAKS_UTILITY_FILE_H

#include <unistd.h>
#include <sys/stat.h>
#include <vector>
#include "path.h"
#include "config_info.h"
#include "umask_guard.h"
#include "ustring.h"

namespace Utility {
    constexpr uint32_t DIRMOD = 0750;
    constexpr uint32_t LEAST_OUTPUT_FILE_MODE = 0775;
    constexpr uint32_t DEFAULT_UMASK_FOR_CSV_FILE = 0177;
    constexpr mode_t FULL_PERMISSIONS = 0777;
    extern std::string g_dirPath;
    constexpr uint64_t MAX_INPUT_FILE_SIZE = 1UL << 33; // 8GB
    constexpr mode_t WRITE_FILE_NOT_PERMITTED = S_IWGRP | S_IWOTH | S_IROTH | S_IXOTH;

    inline void SetDirPath(const std::string& dirPath, const std::string& defaultDirPath)
    {
        if (dirPath.empty()) {
            Utility::Path path = Utility::Path{defaultDirPath};
            Utility::Path realPath = path.Resolved();
            g_dirPath = realPath.ToString();
        } else {
            g_dirPath = dirPath;
        }
    }

    inline bool MakeDir(const std::string& dirPath)
    {
        if (dirPath.empty()) {
            printf("Invalid directory path.\n");
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
                printf("Cannot create dir %s.\n", partPath.c_str());
                return false;
            }
        }

        if (mkdir(dirPath.c_str(), DIRMOD) != 0) {
            printf("Cannot create dir %s.\n", dirPath.c_str());
            return false;
        }
        return true;
    }

    inline bool Exist(const std::string &path)
    {
        if (path.empty()) {
            printf("The file path is empty.");
            return false;
        }
        return access(path.c_str(), F_OK) == 0;
    }

    // 多线程情况下调用，需加锁保护
    bool CreateCsvFile(FILE **filefp, std::string dirPath, std::string fileName, std::string headers);

    // 输入+输出文件专属校验：文件大小
    inline bool IsFileSizeSafe(const std::string& path)
    {
        struct stat buffer;
        if (stat(path.c_str(), &buffer) != 0) {
            printf("Error getting file size for %s:", path.c_str());
            return false;
        }

        if (!S_ISREG(buffer.st_mode)) {
            printf("File %s is not a regular file.", path.c_str());
            return false;
        }

        if (buffer.st_size > MAX_INPUT_FILE_SIZE) {
            printf("File %s exceeds maximum size (%d bytes).", path.c_str(), MAX_INPUT_FILE_SIZE);
            return false;
        }
        return true;
    }

    /// 输出文件校验
    bool CheckFileBeforeCreate(const std::string &path);

    /// 创建文件
    FILE* CreateFileWithUmask(const std::string &path, const std::string &mode, mode_t mask);
    FILE* CreateFile(const std::string &dir, const std::string &name, mode_t mask);

}

#endif