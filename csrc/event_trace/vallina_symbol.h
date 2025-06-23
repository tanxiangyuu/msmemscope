// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef LEAKS_VALLINA_SYMBOL_H
#define LEAKS_VALLINA_SYMBOL_H

#include <string>
#include <array>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>

namespace Leaks {

inline void *LibLoad(std::string libName)
{
    if (libName.empty()) {
        std::cout << "Null library name." << std::endl;
        return nullptr;
    }
    const char *pathEnv = std::getenv("ASCEND_HOME_PATH");
    if (!pathEnv || std::string(pathEnv).empty()) {
        std::cout << "[msleaks] Failed to acquire ASCEND_HOME_PATH environment variable while loading "
            << libName << "."
            << std::endl;
        return nullptr;
    }
    std::string libPath = pathEnv;
    libPath += "/lib64/" + libName;
    return dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
}

static std::string executeCommand(const char* cmd)
{
    std::string result;
    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        std::cerr << "popen() failed!" << std::endl;
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

static std::string getDirname(const std::string& path)
{
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return ".";
    }
    return path.substr(0, pos);
}

static std::string joinPath(const std::string& base, const std::string& component)
{
    if (base.empty()) return component;
    if (base.back() == '/' || base.back() == '\\') {
        return base + component;
    }
    return base + "/" + component;
}

static bool fileExists(const std::string& path)
{
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

static std::string findLibParentDir(const std::string& startPath, int maxDepth = 5)
{
    if (maxDepth <= 0) return "";
    for (const auto& dirName : {"lib", "lib64"}) {
        std::string testPath = joinPath(startPath, dirName);
        if (fileExists(testPath)) {
            return startPath;
        }
    }
    std::string parentPath = getDirname(startPath);
    if (parentPath == startPath) return "";
    return findLibParentDir(parentPath, maxDepth - 1);
}

static bool validateLibrary(const std::string& path)
{
    struct stat st;
    int ret = stat(path.c_str(), &st);
    if (ret != 0) {
        std::cout << "[msleaks] Error getting file status: " << path << std::endl;
        return false;
    }

    // 检查是否为root用户拥有
    if (st.st_uid == 0) return true;

    // 检查是否为当前用户拥有
    uid_t current_uid = getuid();
    if (st.st_uid == current_uid) {
        // 检查是否存在组或其他用户的写权限
        if ((st.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
            std::cout << "[msleaks] Security risk: Library " << path << " is writable by group/others" << std::endl;
            return false;
        }
        return true;
    }
    std::cout << "[msleaks] Security violation: Library " << path
              << " is not owned by root or current user" << std::endl;
    return false;
}

static void* findAndLoadSqliteInDir(const std::string& dirPath, int depth = 0, int maxDepth = 3)
{
    const std::vector<std::string> libCandidates = {
        "libsqlite3.so",
    };
    for (const auto& candidate : libCandidates) {
        std::string candidatePath = joinPath(dirPath, candidate);
        if (fileExists(candidatePath)) {
            bool result = validateLibrary(candidatePath);
            if (!result) {
                continue;
            }
            void* handle = dlopen(candidatePath.c_str(), RTLD_NOW | RTLD_GLOBAL);
            if (handle) return handle;
            std::cout << "[msleaks] Failed to load " << candidatePath << ": " << dlerror() << std::endl;
        }
    }
    DIR* dir = opendir(dirPath.c_str());
    if (!dir) return nullptr;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
        std::string childPath = joinPath(dirPath, entry->d_name);
        struct stat statbuf;
        if (stat(childPath.c_str(), &statbuf) == 0 && S_ISDIR(statbuf.st_mode)) {
            void* handle = findAndLoadSqliteInDir(childPath, depth + 1, maxDepth);
            if (handle) {
                closedir(dir);
                return handle;
            }
        }
    }
    closedir(dir);
    return nullptr;
}

struct Sqlite3LibLoader {
    static void* Load(void)
    {
        std::string whichCmd = "which sqlite3 2>/dev/null";
        std::string sqlite3BinPath = executeCommand(whichCmd.c_str());
        if (!sqlite3BinPath.empty()) {
            std::string libParentDir = findLibParentDir(sqlite3BinPath);
            if (libParentDir.empty()) {
                std::cout << "[msleaks] Could not find lib directory: sqlite3" << std::endl;
                return nullptr;
            }
            for (const auto& dirName : {"lib", "lib64"}) {
                void* handle = findAndLoadSqliteInDir(joinPath(libParentDir, dirName));
                if (handle) {
                    std::cout << "[Warning] Loaded sqlite3 library from: " << libParentDir << std::endl;
                    return handle;
                }
            }
        }
        return nullptr;
    }
};

/* VallinaSymbol 类用于从指定的动态库句柄中获取函数符号
 * @tparam LibLoader 动态库加载器，需要实现 Load 方法
 */
template <typename LibLoader>
class VallinaSymbol {
public:
    inline static VallinaSymbol &Instance(void)
    {
        static VallinaSymbol ins;
        return ins;
    }
    VallinaSymbol(VallinaSymbol const &) = delete;
    VallinaSymbol &operator=(VallinaSymbol const &) = delete;

    /* 获取指定函数名的符号地址
     * @param symbol 要获取的函数符号名
     * @return 获取到的函数符号
     */
    inline void *Get(char const *symbol) const
    {
        if (handle_ == nullptr) {
            std::cout << "[msleaks] lib handle is NULL" << std::endl;
            return nullptr;
        }
        return dlsym(handle_, symbol);
    }

    /* 获取指定函数名的符号地址，并且转换为对应类型的函数指针
     * @param symbol 要获取的函数符号名
     * @return 获取到的函数符号
     */
    template <typename Func>
    inline Func Get(char const *symbol) const
    {
        return reinterpret_cast<Func>(Get(symbol));
    }

private:
    inline VallinaSymbol(void) : handle_(LibLoader::Load())
    {}

private:
    void *handle_;
};
}
#endif
