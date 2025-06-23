// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef LEAKS_UTILITY_USTRING_H
#define LEAKS_UTILITY_USTRING_H

#include <string>
#include <set>
#include <vector>
#include <algorithm>
#include <regex>
#include "utils.h"

namespace Utility {

template <typename Iterator>
inline std::string Join(Iterator beg, Iterator end, std::string const &sep = " ")
{
    std::string ret;
    if (beg == end) {
        return ret;
    }
    ret = *beg++;
    for (; beg != end; ++beg) {
        ret.append(sep);
        ret.append(*beg);
    }
    return ret;
}

/* 字符串分割
 * 使用指定的分隔符对字符串进行分割
 * @param str 要分割的字符串
 * @param it 分割后的字符串保存的容器的迭代器，由调用者保证容器的大小满足要求
 * @param seps 指定的分隔符列表
 * @param strict 是否启用严格模式。严格模式下连续出现的分隔符会被依次处理：
 *        - split "abc::def" with ":" -> ["abc", "", "def"]
 *        非严格模式下连续出现的分隔符会作为一个整体
 *        - split "abc::def" with ":" -> ["abc", "def"]
 */
template <typename Iterator>
inline void Split(std::string const &str, Iterator it, std::string const &seps = " ", bool strict = false)
{
    std::string::size_type slow = 0UL;
    std::string::size_type fast = str.find_first_of(seps);
    *it = str.substr(slow, fast - slow);
    for (; fast < str.length(); ++it) {
        if (strict) {
            slow = fast == std::string::npos || fast + 1 >= str.length() ? std::string::npos : fast + 1UL;
        } else {
            slow = str.find_first_not_of(seps, fast);
        }
        fast = str.find_first_of(seps, slow);
        if (slow == std::string::npos) {
            *it = std::string();
        } else if (fast == std::string::npos) {
            *it = str.substr(slow);
        } else {
            *it = str.substr(slow, fast - slow);
        }
    }
}

inline std::string Strip(std::string str, std::string const &cs = " ")
{
    std::string::size_type l = str.find_first_not_of(cs);
    std::string::size_type r = str.find_last_not_of(cs);
    return l == std::string::npos || r == std::string::npos ? "" : str.substr(l, r - l + 1);
}

inline std::string RStrip(std::string str, std::string const &cs = " ")
{
    std::string::size_type r = str.find_last_not_of(cs);
    return r == std::string::npos ? "" : str.substr(0UL, r + 1);
}

inline bool EndWith(std::string const &str, std::string const &target)
{
    return str.length() >= target.length() && str.substr(str.length() - target.length()) == target;
}

inline bool CheckStrIsStartsWithInvalidChar(const char *const &str)
{
    const std::set<char> invalidCharList = {'=', '+', '-', '@'};
    if (invalidCharList.count(*str) != 0) {
        return true;
    }
    return false;
}

std::string ExtractAttrValueByKey(const std::string& str, const std::string& key);

// 去除空格
inline std::string Trim(const char* str)
{
    const char* start = str;
    const char* end = start;
    while (*end != '\0') {
        end++;
    }
    return std::string(start, end);
}

class Version {
public:
    Version() = delete;
    // ver的版本号以'.'分隔，形式为x.y.z....
    explicit Version(const std::string &ver)
    {
        std::regex numberPattern(R"(^(0|[1-9]\d*)$)");
        std::vector<std::string> version;
        Split(ver, std::back_inserter(version), ".");
        for (auto s : version) {
            uint32_t versionNum;
            if (s != "" && std::regex_match(s, numberPattern) && StrToUint32(versionNum, s)) {
                version_.push_back(versionNum);
            }
        }
    }

    bool operator<(const Version &other)
    {
        size_t size = std::min(version_.size(), other.version_.size());
        for (size_t i = 0; i < size; ++i) {
            if (version_[i] != other.version_[i]) {
                return version_[i] < other.version_[i];
            }
        }
        return version_.size() < other.version_.size();
    }

    bool operator>(const Version &other)
    {
        size_t size = std::min(version_.size(), other.version_.size());
        for (size_t i = 0; i < size; ++i) {
            if (version_[i] != other.version_[i]) {
                return version_[i] > other.version_[i];
            }
        }
        return version_.size() > other.version_.size();
    }

    bool operator==(const Version &other)
    {
        size_t size = std::min(version_.size(), other.version_.size());
        for (size_t i = 0; i < size; ++i) {
            if (version_[i] != other.version_[i]) {
                return false;
            }
        }
        return version_.size() == other.version_.size();
    }

    bool operator>=(const Version &other)
    {
        return *this > other || *this == other;
    }

    bool operator<=(const Version &other)
    {
        return *this < other || *this == other;
    }

private:
    std::vector<uint32_t> version_;
};

}  // namespace Utility

#endif