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

void *LibLoadFromDriver(std::string libName);
std::string ExecuteCommand(const char* cmd);
std::string GetDirname(const std::string& path);
std::string JoinPath(const std::string& base, const std::string& component);
bool FileExists(const std::string& path);
std::string FindLibParentDir(const std::string& startPath, int maxDepth = 5);
bool ValidateLibrary(const std::string& path);
void* FindAndLoadSqliteInDir(const std::string& dirPath, int depth = 0, int maxDepth = 3);

struct Sqlite3LibLoader {
    static void* Load(void)
    {
        std::string whichCmd = "which sqlite3 2>/dev/null";
        std::string sqlite3BinPath = ExecuteCommand(whichCmd.c_str());
        if (!sqlite3BinPath.empty()) {
            std::string libParentDir = FindLibParentDir(sqlite3BinPath);
            if (libParentDir.empty()) {
                std::cout << "[msleaks] Could not find lib directory: sqlite3" << std::endl;
                return nullptr;
            }
            for (const auto& dirName : {"lib", "lib64"}) {
                void* handle = FindAndLoadSqliteInDir(JoinPath(libParentDir, dirName));
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
