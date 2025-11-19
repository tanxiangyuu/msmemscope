// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <vector>
#include <cstring>
#include "vallina_symbol.h"

namespace MemScope {
    void *LibLoad(std::string libName)
    {
        if (libName.empty()) {
            std::cout << "[msmemscope] Error: Null library name." << std::endl;
            return nullptr;
        }
        std::string libPath = libName;
        const char *pathEnv = std::getenv("ASCEND_HOME_PATH");
        if (pathEnv && !std::string(pathEnv).empty()) {
            libPath = pathEnv;
            libPath += "/lib64/" + libName;
            return dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
        }
        // 找不到Ascend Path
        std::cout << "[msmemscope] Error: Failed to acquire ASCEND_HOME_PATH environment variable while loading "
            << libName << ". Try to load lib directly."
            << std::endl;
        return dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    }
    
    void *GetSymbol(char const *symbol)
    {
        void *func = dlsym(nullptr, symbol);
        return func;
    }

    std::string ExecuteCommand(const char* cmd)
    {
        std::string result;
        FILE* pipe = popen(cmd, "r");
        if (!pipe) {
            std::cout << "[msmemscope] Error: popen() failed!" << std::endl;
            return result;
        }
        int bufferSize = 128;
        std::vector<char> buffer(bufferSize);
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            result += buffer.data();
        }
        pclose(pipe);
        if (!result.empty() && result.back() == '\n') {
            result.pop_back();
        }
        return result;
    }

    std::string GetDirname(const std::string& path)
    {
        size_t pos = path.find_last_of("/\\");
        if (pos == std::string::npos) {
            return ".";
        }
        return path.substr(0, pos);
    }

    std::string JoinPath(const std::string& base, const std::string& component)
    {
        if (base.empty()) {
            return component;
        }

        if (base.back() == '/' || base.back() == '\\') {
            return base + component;
        }
        return base + "/" + component;
    }

    bool FileExists(const std::string& path)
    {
        struct stat buffer;
        return (stat(path.c_str(), &buffer) == 0);
    }

    std::string FindLibParentDir(const std::string& startPath, int maxDepth)
    {
        if (maxDepth <= 0) {
            return "";
        }
        for (const auto& dirName : {"lib", "lib64"}) {
            std::string testPath = JoinPath(startPath, dirName);
            if (FileExists(testPath)) {
                return startPath;
            }
        }
        std::string parentPath = GetDirname(startPath);
        if (parentPath == startPath) {
            return "";
        }
        return FindLibParentDir(parentPath, maxDepth - 1);
    }

    bool ValidateLibrary(const std::string& path)
    {
        struct stat st;
        int ret = stat(path.c_str(), &st);
        if (ret != 0) {
            std::cout << "[msmemscope] Error: getting file status: " << path << std::endl;
            return false;
        }

        // 检查是否为root用户拥有
        if (st.st_uid == 0) {
            return true;
        }

        // 检查是否为当前用户拥有
        uid_t currentUid = getuid();
        if (st.st_uid == currentUid) {
            // 检查是否存在组或其他用户的写权限
            if ((st.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
                std::cout << "[msmemscope] Security risk: Library " << path << " is writable by group/others" << std::endl;
                return false;
            }
            return true;
        }
        std::cout << "[msmemscope] Security violation: Library " << path
                << " is not owned by root or current user" << std::endl;
        return false;
    }

    void* FindAndLoadSqliteInDir(const std::string& dirPath, int depth, int maxDepth)
    {
        if (depth > maxDepth) {
            return nullptr;
        }
        const std::vector<std::string> libCandidates = {
            "libsqlite3.so",
        };
        for (const auto& candidate : libCandidates) {
            std::string candidatePath = JoinPath(dirPath, candidate);
            if (FileExists(candidatePath)) {
                bool result = ValidateLibrary(candidatePath);
                if (!result) {
                    continue;
                }
                void* handle = dlopen(candidatePath.c_str(), RTLD_NOW | RTLD_GLOBAL);
                if (handle) {
                    return handle;
                }
                std::cout << "[msmemscope] Error: Failed to load " << candidatePath << ": " << dlerror() << std::endl;
            }
        }
        DIR* dir = opendir(dirPath.c_str());
        if (!dir) {
            return nullptr;
        }
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
                continue;
            }
            std::string childPath = JoinPath(dirPath, entry->d_name);
            struct stat statbuf;
            if (stat(childPath.c_str(), &statbuf) == 0 && S_ISDIR(statbuf.st_mode)) {
                void* handle = FindAndLoadSqliteInDir(childPath, depth + 1, maxDepth);
                if (handle) {
                    closedir(dir);
                    return handle;
                }
            }
        }
        closedir(dir);
        return nullptr;
    }
}
