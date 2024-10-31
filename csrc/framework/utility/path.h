// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef LEAKS_UTILITY_PATH_H
#define LEAKS_UTILITY_PATH_H

#include <string>
#include <vector>
#include <sys/stat.h>

namespace Leaks {

class Path {
public:
    Path(void) noexcept;
    explicit Path(std::string path) noexcept;
    Path(Path const &) = default;
    Path(Path &&) = default;
    Path &operator=(Path const &) = default;
    Path &operator=(Path &&) = default;

    /// 将 Path 对象拼接为原始路径字符串
    std::string ToString(void) const;

    /// 获取路径中最后一个文件或目录名
    std::string Name(void) const;

    /// 获取父路径
    Path Parent(void) const &;
    Path Parent(void) &&;

    /// 获取绝对路径
    Path Absolute(void) const &;
    Path Absolute(void) &&;

    /// 路径正规化
    Path Resolved(void) const &;
    Path Resolved(void) &&;

    /// 路径拼接
    Path operator/(Path rhs) const &;
    Path operator/(Path rhs) &&;

    /// 获取路径文件状态
    bool GetStat(struct stat &st) const;

    /// 校验路径是否存在
    bool Exists(void) const;

private:
    bool absolute_;
    std::vector<std::string> route_;
};

}  // namespace Leaks

#endif  // !CORE_FRAMEWORK_UTILITY_PATH_H
