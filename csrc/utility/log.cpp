// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include "log.h"
#include <chrono>
#include <ctime>
#include <type_traits>
#include <map>
#include "file.h"

namespace Utility {
const int FILEPOSITION = 2;

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
        fp_ = nullptr;
    }
}
std::string Log::AddPrefixInfo(std::string const &format, LogLv lv, const std::string fileName,
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
bool Log::CreateLogFile()
{
    if (fp_ == nullptr) {
        std::string fileName = "msleaks_" + GetDateStr() + ".log";
        logFilePath_ = "./" + fileName;
        if ((fp_ = CreateFile(".", fileName, DEFAULT_UMASK_FOR_LOG_FILE)) == nullptr) {
            return false;
        }
        std::cout << "[msleaks] Info: logging into file " << fileName << std::endl;
    }
    return true;
}
void Log::SetLogLevel(const LogLv &logLevel)
{
    lv_ = logLevel;
}

void Log::RotateLogFile()
{
    if (fp_ == nullptr) {
        return;
    }
    fclose(fp_);
    fp_ = nullptr;
    if (rotateCount_ >= MAX_LOG_FILE_NUMBER) {
        printf("[msleaks] the number of rotated log files exceeded limit(%ld), remove oldest log.\n",
               MAX_LOG_FILE_NUMBER);
        std::string oldestFile = logFilePath_ + ".bak." + std::to_string(MAX_LOG_FILE_NUMBER - 1);
        if (!IsRegFile(oldestFile)) {
            printf("[msleaks] Warn: %s is not a regular file\n", oldestFile.c_str());
            return;
        }
        if (remove(oldestFile.c_str()) != 0) {
            printf("[msleaks] failed to remove bak log file: %s\n", oldestFile.c_str());
            return;
        }
    }
    for (auto idx = std::min(rotateCount_, MAX_LOG_FILE_NUMBER - 1); idx > 0; idx--) {
        std::string oldPath = idx != 1 ? logFilePath_ + ".bak." + std::to_string(idx - 1) : logFilePath_;
        std::string newPath = logFilePath_ + ".bak." + std::to_string(idx);
        if (!Utility::Exist(oldPath) || Utility::Exist(newPath)) {
            continue;
        }
        if (rename(oldPath.c_str(), newPath.c_str()) != 0) {
            printf("[msleaks] failed to rotate bak log file.\n");
            return;
        }
    }
    fp_ = CreateFile(".", logFilePath_.substr(FILEPOSITION), DEFAULT_UMASK_FOR_LOG_FILE);
    if (fp_ != nullptr) {
        rotateCount_++;
    } else {
        printf("[msleaks] failed to open log file: %s\n", logFilePath_.c_str());
    }
}

inline std::string GetLogSourceFileName(const std::string &path);

}  // namespace Utility
