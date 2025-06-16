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
}  // namespace Utility