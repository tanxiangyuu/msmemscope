// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "file.h"
#include "utils.h"

namespace Utility {

    std::string g_dirPath;

    // Log创建之前不要使用LOG_***
    bool CheckFileBeforeCreate(const std::string &path)
    {
        Path curPath = Path(path).Absolute();
        std::string curPathStr = curPath.ToString();
        std::string users;

        if (!curPath.Exists()) {
            printf("The path %s not exists", curPathStr.c_str());
            return false;
        }
        if (!CheckStrIsStartsWithInvalidChar(curPathStr.c_str())) {
            printf("The path %s is invalid", curPathStr.c_str());
            return false;
        }
        if (!curPath.IsReadable()) {
            printf("The file path %s is not readable.", curPathStr.c_str());
            return false;
        }
        if (!curPath.IsValidLength()) {
            printf("The length of file path %s exceeds the maximum length.", curPathStr.c_str());
            return false;
        }
        if (!curPath.IsValidDepth()) {
            printf("The depth of file path %s exceeds the maximum depth.", curPathStr.c_str());
            return false;
        }
        if (curPath.IsSoftLink()) {
            printf("The file path %s is invalid: soft link is not allowed.", curPathStr.c_str());
            return false;
        }
        if (!curPath.IsPermissionValid()) {
            printf("The file path %s is invalid: permission is not valid.", curPathStr.c_str());
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
                fprintf(fp, headers.c_str());
                *filefp = fp;
            } else {
                printf("open file %s error", filePath.c_str());
                return false;
            }
        }
        return true;
    }
}  // namespace Utility