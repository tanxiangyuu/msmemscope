// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#ifndef STRING_VALIDATOR_H
#define STRING_VALIDATOR_H

#include <vector>
#include <string>
#include <cstdint>

namespace Utility {
// 整数验证规则配置（按需扩展字段）
struct IntValidateRule {
    uint32_t minValue = 0;    // 最小允许值（默认0：非负整数）
    uint32_t maxValue = UINT32_MAX;  // 最大允许值（默认无上限）目前最大
    bool allowLeadingZero = false;   // 是否允许前导0（如 "0123"，默认不允许）
};

// 分割字符串（支持多分隔符）
std::vector<std::string> SplitString(const std::string &str, const std::string &delimiters);

// 只允许\\.|/|_|-|\\s|[~0-9a-zA-Z]和中文为有效输出路径字符
bool IsValidOutputPathChar(const char c);
// 检查中文，只允许三字节中文
bool IsChineseCharacter(const unsigned char* utf8);
bool IsValidOutputPath(const std::string &pathStr);

// 验证数据跟踪级别合法性（替代正则：^(0|1|op|kernel)$）
bool IsValidDataLevel(const std::string &level);

// 检查数字是否合法，包括整数限制，是否允许前导0.
bool IsValidInteger(const std::string &str, const IntValidateRule &rule = {});
}
#endif