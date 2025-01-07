// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef LEAKS_UTILITY_FILE_H
#define LEAKS_UTILITY_FILE_H

#include <unistd.h>
#include <sys/stat.h>
#include "log.h"

namespace Utility {
    constexpr uint32_t DIRMOD = 0777;

    static bool MakeDir(const std::string& dirPath)
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
}

#endif