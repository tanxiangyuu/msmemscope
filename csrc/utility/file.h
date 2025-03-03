// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef LEAKS_UTILITY_FILE_H
#define LEAKS_UTILITY_FILE_H

#include <unistd.h>
#include <sys/stat.h>
#include <vector>
#include "log.h"
#include "path.h"
#include "config_info.h"
#include "umask_guard.h"

namespace Utility {
    constexpr uint32_t DIRMOD = 0750;
    constexpr uint32_t DEFAULT_UMASK_FOR_CSV_FILE = 0177;
    extern std::string g_dirPath;

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
            LOG_ERROR("Invalid directory path.");
            return false;
        }

        if (access(dirPath.c_str(), F_OK) != -1) {
            return true;
        }
        LOG_INFO("dir %s does not exist", dirPath.c_str());
        
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
                LOG_ERROR("Cannot create dir %s", partPath.c_str());
                return false;
            }
        }

        if (mkdir(dirPath.c_str(), DIRMOD) != 0) {
            LOG_ERROR("Cannot create dir %s", dirPath.c_str());
            return false;
        }
        return true;
    }

    inline bool Exist(const std::string &path)
    {
        if (path.empty()) {
            LOG_INFO("The file path is empty.");
            return false;
        }
        return access(path.c_str(), F_OK) == 0;
    }

    // 多线程情况下调用，需加锁保护
    inline bool CreateCsvFile(FILE **filefp, std::string dirPath, std::string fileName, std::string headers)
    {
        if (!MakeDir(dirPath)) {
            return false;
        }
        if (*filefp == nullptr) {
            fileName = fileName + Utility::GetDateStr() + ".csv";
            std::string filePath = dirPath + "/" + fileName;
            UmaskGuard guard{DEFAULT_UMASK_FOR_CSV_FILE};
            FILE* fp = fopen(filePath.c_str(), "a");
            if (fp != nullptr) {
                std::cout << "[msleaks] Info: create file " << filePath << "." << std::endl;
                fprintf(fp, headers.c_str());
                *filefp = fp;
            } else {
                LOG_ERROR("open file %s error", filePath.c_str());
                return false;
            }
        }
        return true;
    }
}

#endif