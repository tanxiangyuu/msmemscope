/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */
#include "log.h"
#include <chrono>
#include <ctime>
#include <type_traits>
#include <map>
#include "file.h"
#include "json_manager.h"
#include "trace_manager/event_trace_manager.h"

namespace Utility {

inline std::string ToString(MemScope::LogLv lv)
{
    using underlying = typename std::underlying_type<MemScope::LogLv>::type;
    constexpr char const *lvString[static_cast<underlying>(MemScope::LogLv::COUNT)] = {
        "[DEBUG]", "[INFO] ", "[WARN] ", "[ERROR]"};
    return lv < MemScope::LogLv::COUNT ? lvString[static_cast<underlying>(lv)] : "N";
}

Log &Log::GetLog(void)
{
    static Log instance;
    return instance;
}

Log::Log(void)
{
    MemScope::Config config;
    config = MemScope::GetConfig();
    outputDir_ = config.outputDir;
}

Log::~Log()
{
    if (fp_ != nullptr) {
        fclose(fp_);
        fp_ = nullptr;
    }
}
std::string Log::AddPrefixInfo(std::string const &format, MemScope::LogLv lv, const std::string fileName,
    const uint32_t line) const
{
    char buf[LOG_BUF_SIZE];
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm *tm = std::localtime(&time);
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm);
    std::string codePosition = "[" + fileName + ":" + std::to_string(line) + "] ";
    return std::string(buf) + " " + ToString(lv) + " " + codePosition + format;
}
void Log::SetLogLevel(const MemScope::LogLv &logLevel)
{
    lv_ = logLevel;
}

inline std::string GetLogSourceFileName(const std::string &path);

}  // namespace Utility
