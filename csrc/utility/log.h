// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#ifndef LEAKS_UTILITY_LOG_H
#define LEAKS_UTILITY_LOG_H

#include <type_traits>
#include <string>
#include <mutex>
#include "utils.h"
#include "umask_guard.h"
#include "cstring"
#include "config_info.h"

namespace Utility {

constexpr uint16_t LOG_BUF_SIZE = 32;
constexpr uint32_t DEFAULT_UMASK_FOR_LOG_FILE = 0177;
constexpr int16_t DOUBLE = 2;

inline bool operator<(Leaks::LogLv a, Leaks::LogLv b)
{
    using underlying = typename std::underlying_type<Leaks::LogLv>::type;
    return static_cast<underlying>(a) < static_cast<underlying>(b);
}

class Log {
public:
    static Log &GetLog(void);

    template <typename... Args>
    inline void Printf(const std::string &format, Leaks::LogLv lv, const std::string fileName, const uint32_t line,
        const Args& ...args);
    template <typename... Args>
    inline void Printtest(const std::string &format, Leaks::LogLv lv, const std::string fileName, const uint32_t line,
        const Args& ...args);
    template <typename... Args>
    inline void PrintClientLog(std::string const &format, const Args &...args);
    void SetLogLevel(const Leaks::LogLv &logLevel);
    bool CreateLogFile();
private:
    Log(void) = default;
    ~Log(void);
    Log(Log const &) = delete;
    Log &operator=(Log const &) = delete;
    std::string AddPrefixInfo(std::string const &format, Leaks::LogLv lv, const std::string fileName,
        const uint32_t line) const;
    inline int64_t LogSize() const
    {
        if (fp_ == nullptr) {
            return 0;
        }
        int rt = fseeko(fp_, 0L, SEEK_END);
        if (rt != 0) {
            return -1;
        }
        int64_t size = ftello64(fp_);
        return size;
    }

private:
    Leaks::LogLv lv_{Leaks::LogLv::WARN};
    FILE *fp_{nullptr};
    mutable std::mutex mtx_;
    int64_t maxLogSize_ = 100L * 1024L * 1024L; // 100M
    std::string logFilePath_;
};

template <typename... Args>
void Log::Printf(const std::string &format, Leaks::LogLv lv, const std::string fileName, const uint32_t line,
    const Args& ...args)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (!CreateLogFile()) {
        return;
    }
    if (lv < lv_) {
        return;
    }
    std::string f = AddPrefixInfo(format, lv, fileName, line).append("\n");
    if (LogSize() + static_cast<int64_t>(f.size()) > maxLogSize_) {
        std::cout << "[msleaks] Warn: Log file size is too large, please check: " << logFilePath_ << std::endl;
        maxLogSize_ *= DOUBLE;
    }
    if (fp_ != nullptr) {
        fprintf(fp_, f.c_str(), args...);
        fflush(fp_);
    } else {
        std::cout << "[msleaks] Error: open file " << logFilePath_ << " failed." << std::endl;
    }
}

template <typename... Args>
void Log::PrintClientLog(const std::string &format, const Args &...args)
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
    if (LogSize() + static_cast<int64_t>(f.size()) > maxLogSize_) {
        std::cout << "[msleaks] Warn: Log file size is too large, please check: " << logFilePath_ << std::endl;
        maxLogSize_ *= DOUBLE;
    }
    if (fp_ != nullptr) {
        fprintf(fp_, f.c_str(), args...);
        fflush(fp_);
    } else {
        std::cout << "[msleaks] Error: open file " << logFilePath_ << " failed." << std::endl;
    }
}

inline std::string GetLogSourceFileName(const std::string &path)
{
    return (strrchr(path.c_str(), '/')) ? (strrchr(path.c_str(), '/') + 1) : path;
}

#define LOG_RECV(format, ...)                                                                                          \
    do {                                                                                                               \
        Utility::Log::GetLog().PrintClientLog(format, ##__VA_ARGS__);                                                  \
    } while (0)

#define LOG_DEBUG(format, ...)                                                                                         \
    do {                                                                                                               \
        Utility::Log::GetLog().Printf(format, Leaks::LogLv::DEBUG, Utility::GetLogSourceFileName(__FILE__),            \
        __LINE__, ##__VA_ARGS__);                                                                                      \
    } while (0)

#define LOG_INFO(format, ...)                                                                                          \
    do {                                                                                                               \
        Utility::Log::GetLog().Printf(format, Leaks::LogLv::INFO, Utility::GetLogSourceFileName(__FILE__),             \
        __LINE__, ##__VA_ARGS__);                                                                                      \
    } while (0)

#define LOG_WARN(format, ...)                                                                                          \
    do {                                                                                                               \
        Utility::Log::GetLog().Printf(format, Leaks::LogLv::WARN, Utility::GetLogSourceFileName(__FILE__),             \
        __LINE__, ##__VA_ARGS__);                                                                                      \
    } while (0)

#define LOG_ERROR(format, ...)                                                                                         \
    do {                                                                                                               \
        Utility::Log::GetLog().Printf(format, Leaks::LogLv::ERROR, Utility::GetLogSourceFileName(__FILE__),            \
        __LINE__, ##__VA_ARGS__);                                                                                      \
    } while (0)

inline void SetLogLevel(const Leaks::LogLv &logLevel)
{
    Log::GetLog().SetLogLevel(logLevel);
}
}  // namespace Utility

#endif