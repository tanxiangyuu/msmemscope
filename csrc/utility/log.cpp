// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "log.h"
#include <chrono>
#include <ctime>
#include <type_traits>
#include <map>

namespace Utility {

inline std::string ToString(LogLv lv)
{
    using underlying = typename std::underlying_type<LogLv>::type;
    constexpr char const *lvString[static_cast<underlying>(LogLv::COUNT)] = {
        "[DEBUG]", "[INFO] ", "[WARN] ", "[ERROR]"};
    return lv < LogLv::COUNT ? lvString[static_cast<underlying>(lv)] : "N";
}

Log &Log::GetLog(void)
{
    static Log instance;
    return instance;
}
Log::~Log()
{
    if (fp_ != nullptr) {
        fclose(fp_);
    }
}
std::string Log::AddPrefixInfo(std::string const &format, LogLv lv) const
{
    char buf[LOG_BUF_SIZE];
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm *tm = std::localtime(&time);
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm);
    return std::string(buf) + " " + ToString(lv) + " " + format;
}

void Log::SetLogLevel(const LogLv &logLevel)
{
    lv_ = logLevel;
}

}  // namespace Utility
