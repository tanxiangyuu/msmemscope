// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "path.h"

#include <unistd.h>
#include <sys/stat.h>
#include <linux/limits.h>

#include "ustring.h"

namespace Utility {

constexpr char const *PATH_SEP = "/";
constexpr char const *CURRENT_SEGMENT = ".";
constexpr char const *PARENT_SEGMENT = "..";

Path::Path(void) noexcept : absolute_{false}
{}

/* Path 构造函数
 * 构造函数中的核心逻辑为将一个路径字符串解析为一系列由 PATH_SEP 分隔的构件（components）组成，
 * 由 std::vector<std::string> 表示。路径可分为相对路径和绝对路径。相对路径在 Path 中表示为第一
 * 个构件不以分隔符开头，如 `["test", "test.cpp"]`；绝对路径的第一个构件以分隔符开头，如
 * `["/test", "test.cpp"]`。
 */
Path::Path(std::string path) noexcept : absolute_{false}
{
    path = Strip(path);
    /// 空路径表示当前路径 "."，并以空列表表示
    if (path.empty()) {
        return;
    }

    std::string route;
    std::string::size_type slow = 0UL;
    std::string::size_type fast = path.find_first_of(PATH_SEP, slow);
    /// 如果路径以分隔符开头需要特殊处理开头的部分
    if (fast == slow) {
        std::string::size_type next = path.find_first_not_of(PATH_SEP, fast);
        fast = path.find_first_of(PATH_SEP, next);
        absolute_ = true;
        /// 如果 next == std::string::npos 说明路径由若干分隔符组成，处理成 ["/"]
        if (next == std::string::npos) {
            return;
        } else if (fast == std::string::npos) {
            route = path.substr(next);
        } else {
            route = path.substr(next, fast - next);
        }
        route_.emplace_back(route);
        slow = path.find_first_not_of(PATH_SEP, fast);
    }

    for (; slow != std::string::npos;) {
        fast = path.find_first_of(PATH_SEP, slow);
        route = fast == std::string::npos ? path.substr(slow) : path.substr(slow, fast - slow);
        if (route != CURRENT_SEGMENT) {
            route_.emplace_back(route);
        }
        slow = path.find_first_not_of(PATH_SEP, fast);
    }
}

std::string Path::ToString(void) const
{
    std::string raw;
    /// 空列表表示当前路径 "."
    if (absolute_) {
        raw = PATH_SEP;
    } else if (route_.empty()) {
        return CURRENT_SEGMENT;
    }

    raw.append(Join(route_.cbegin(), route_.cend(), PATH_SEP));
    return raw;
}

std::string Path::Name(void) const
{
    if (route_.empty()) {
        return "";
    }
    return route_.back();
}

Path Path::Parent(void) const &
{
    return Path(*this).Parent();
}

Path Path::Parent(void) &&
{
    if (!route_.empty()) {
        route_.pop_back();
    }
    return std::move(*this);
}

Path Path::Absolute(void) const &
{
    return Path(*this).Absolute();
}

Path Path::Absolute(void) &&
{
    char buf[PATH_MAX] = {0};
    std::string cwd;
    if (getcwd(buf, sizeof(buf))) {
        cwd = buf;
    }
    return Path(cwd) / std::move(*this);
}

Path Path::Resolved(void) const &
{
    return Path(*this).Resolved();
}

Path Path::Resolved(void) &&
{
    Path path = std::move(*this).Absolute();
    auto fast = path.route_.cbegin();
    auto slow = path.route_.begin();
    for (; fast != path.route_.cend(); ++fast) {
        if (*fast != PARENT_SEGMENT) {
            *slow++ = *fast;
        } else if (slow > path.route_.begin()) {
            --slow;
        }
    }
    path.route_.erase(slow, path.route_.end());
    return path;
}

Path Path::operator/(Path rhs) const &
{
    return Path(*this) / rhs;
}

Path Path::operator/(Path rhs) &&
{
    if (rhs.absolute_) {
        return std::move(rhs);
    }

    for (auto &r : rhs.route_) {
        route_.emplace_back(std::move(r));
    }
    return std::move(*this);
}

bool Path::GetStat(struct stat &st) const
{
    return stat(this->ToString().c_str(), &st) == 0;
}

bool Path::Exists(void) const
{
    struct stat st {};
    return stat(this->ToString().c_str(), &st) == 0;
}

bool Path::IsValidLength(void) const
{
    std::size_t pathNameLength = 0;
    for (auto it = this->route_.cbegin(); it != this->route_.cend(); ++it) {
        std::size_t fileNameLength = it->length();
        if (fileNameLength > NAME_MAX || fileNameLength == 0) {
            return false;
        }
        pathNameLength += fileNameLength;
    }
    if (pathNameLength > PATH_MAX || pathNameLength == 0) {
        return false;
    }
    return true;
}

}  // namespace Utility
