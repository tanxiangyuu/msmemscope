// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef LEAKS_UTILITY_FILE_H
#define LEAKS_UTILITY_FILE_H

#include <unistd.h>
#include <sys/stat.h>
#include "log.h"
#include "umask_guard.h"

namespace Utility {
    constexpr uint32_t DIRMOD = 0750;
    constexpr uint32_t DEFAULT_UMASK_FOR_CSV_FILE = 0177;

    inline bool MakeDir(const std::string& dirPath)
    {
        if (access(dirPath.c_str(), F_OK) == -1) {
            Utility::LogInfo("dir %s does not exist", dirPath.c_str());
            if (mkdir(dirPath.c_str(), DIRMOD) != 0) {
                Utility::LogError("cannot create dir %s", dirPath.c_str());
                return false;
            }
        }
        return true;
    }

    inline bool Exist(const std::string &path)
    {
        if (path.empty()) {
            Utility::LogInfo("The file path is empty.");
            return false;
        }
        return access(path.c_str(), F_OK) == 0;
    }

    inline bool CreateCsvFile(FILE **filefp, std::string dirPath, std::string fileName, std::string headers)
    {
        if (!MakeDir(dirPath)) {
            return false;
        }
        if (*filefp == nullptr) {
            std::string filePath = dirPath + "/" + fileName;
            UmaskGuard guard{DEFAULT_UMASK_FOR_CSV_FILE};
            FILE* fp = fopen(filePath.c_str(), "a");
            if (fp != nullptr) {
                fprintf(fp, headers.c_str());
                *filefp = fp;
            } else {
                Utility::LogError("open file %s error", filePath.c_str());
                return false;
            }
        }
        return true;
    }
}

#endif