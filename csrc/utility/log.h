// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#ifndef LEAKS_UTILITY_LOG_H
#define LEAKS_UTILITY_LOG_H

#include <type_traits>
#include <string>

namespace Utility {

constexpr int16_t LOG_BUF_SIZE = 32;

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
    inline void Printf(std::string const &format, LogLv lv, Args &&...args) const;
    void SetLogLevel(const std::string &logLevel = "1");

private:
    Log(void) = default;
    ~Log(void) = default;
    Log(Log const &) = delete;
    Log &operator=(Log const &) = delete;
    std::string AddPrefixInfo(std::string const &format, LogLv lv) const;

private:
    LogLv lv_{LogLv::INFO};
    FILE *fp_{stdout};
};

template <typename... Args>
void Log::Printf(const std::string &format, LogLv lv, Args &&...args) const
{
    if (fp_ == nullptr) {
        return;
    }
    if (lv < lv_) {
        return;
    }
    std::string f = AddPrefixInfo(format, lv).append("\n");
    fprintf(fp_, f.c_str(), std::forward<Args>(args)...);
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

inline void SetLogLevel(const std::string &logLevel)
{
    Log::GetLog().SetLogLevel(logLevel);
}
}  // namespace Utility

#endif