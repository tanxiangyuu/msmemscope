// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#ifndef LEAKS_UTILITY_LOG_H
#define LEAKS_UTILITY_LOG_H

#include <type_traits>
#include <string>
#include <mutex>
#include "utils.h"
#include "umask_guard.h"
#include "path.h"

namespace Utility {

constexpr int16_t LOG_BUF_SIZE = 32;
constexpr uint32_t DEFAULT_UMASK_FOR_LOG_FILE = 0177;

enum class LogLv { DEBUG = 0, INFO, WARN, ERROR, COUNT };

inline bool operator<(LogLv a, LogLv b)
{
    using underlying = typename std::underlying_type<LogLv>::type;
    return static_cast<underlying>(a) < static_cast<underlying>(b);
}

class Log {
public:
    static Log &GetLog(void);

    template <typename... Args>
    inline void Printf(std::string const &format, LogLv lv, Args &&...args);
    template <typename... Args>
    inline void PrintClientLog(std::string const &format, Args &&...args);
    void SetLogLevel(const LogLv &logLevel);
    inline bool CreateLogFile();
private:
    Log(void) = default;
    ~Log(void);
    Log(Log const &) = delete;
    Log &operator=(Log const &) = delete;
    std::string AddPrefixInfo(std::string const &format, LogLv lv) const;

private:
    LogLv lv_{LogLv::WARN};
    FILE *fp_{nullptr};
    mutable std::mutex mtx_;
};

bool Log::CreateLogFile()
{
    if (fp_ == nullptr) {
        std::string fileName = "msleaks_" + GetDateStr() + ".log";
        UmaskGuard guard{DEFAULT_UMASK_FOR_LOG_FILE};

        // 校验路径合法性
        if (!CheckIsValidPath(fileName)) {
            std::cerr << "Error: Invalid path " << fileName << std::endl;
            return false;
        }

        if ((fp_ = fopen(fileName.c_str(), "w")) == nullptr) {
            return false;
        }
        std::cout << "[msleaks] Info: logging into file ./" << fileName << std::endl;
    }
    return true;
}

template <typename... Args>
void Log::Printf(const std::string &format, LogLv lv, Args &&...args)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (!CreateLogFile()) {
        return;
    }
    if (lv < lv_) {
        return;
    }
    std::string f = AddPrefixInfo(format, lv).append("\n");
    fprintf(fp_, f.c_str(), std::forward<Args>(args)...);
    fflush(fp_);
}

template <typename... Args>
void Log::PrintClientLog(const std::string &format, Args &&...args)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (!CreateLogFile()) {
        return;
    }
    char buf[LOG_BUF_SIZE];
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm *tm = std::localtime(&time);
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm);
    std::string f = std::string(std::string(buf) + " " + format).append("\n");
    fprintf(fp_, f.c_str(), std::forward<Args>(args)...);
    fflush(fp_);
}

template <typename... Args>
inline void LogRecv(std::string const &format, Args &&...args)
{
    Log::GetLog().PrintClientLog(format, std::forward<Args>(args)...);
}

template <typename... Args>
inline void LogDebug(std::string const &format, Args &&...args)
{
    Log::GetLog().Printf(format, LogLv::DEBUG, std::forward<Args>(args)...);
}

template <typename... Args>
inline void LogInfo(std::string const &format, Args &&...args)
{
    Log::GetLog().Printf(format, LogLv::INFO, std::forward<Args>(args)...);
}

template <typename... Args>
inline void LogWarn(std::string const &format, Args &&...args)
{
    Log::GetLog().Printf(format, LogLv::WARN, std::forward<Args>(args)...);
}

template <typename... Args>
inline void LogError(std::string const &format, Args &&...args)
{
    Log::GetLog().Printf(format, LogLv::ERROR, std::forward<Args>(args)...);
}

inline void SetLogLevel(const LogLv &logLevel)
{
    Log::GetLog().SetLogLevel(logLevel);
}
}  // namespace Utility

#endif