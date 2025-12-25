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

#include <cstdint>
#include "utils.h"
#include "string_validator.h"

namespace Utility {

std::vector<std::string> SplitString(const std::string &str, const std::string &delimiters)
{
    std::vector<std::string> tokens;
    size_t pos = 0;
    
    while (pos < str.length()) {
        size_t next = str.find_first_of(delimiters, pos);
        if (next == std::string::npos) {
            tokens.push_back(str.substr(pos));
            break;
        }
        tokens.push_back(str.substr(pos, next - pos));
        pos = next + 1;
    }
    
    return tokens;
}

bool IsValidOutputPathChar(const char c)
{
    if (c == '.' || c == '/' || c == '_' || c == '-' || std::isspace(static_cast<unsigned char>(c)) || c == '~') {
        return true;
    }
    if (std::isalnum(static_cast<unsigned char>(c))) {
        return true;
    }
    return false;
}

bool IsChineseCharacter(const unsigned char* utf8)
{
    unsigned char b0 = utf8[0];
    unsigned char b1 = utf8[1];
    unsigned char b2 = utf8[2];

    // 必须是三字节格式：1110xxxx 10xxxxxx 10xxxxxx
    if ((b0 & 0xF0) != 0xE0 || (b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80) {
        return false;
    }

    // 组合成 Unicode 码点
    uint32_t codepoint = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);

    // 只允许基本汉字：U+4E00 ~ U+9FA5
    return (codepoint >= 0x4E00 && codepoint <= 0x9FA5);
}

bool IsValidOutputPath(const std::string &pathStr)
{
    if (pathStr.empty()) {
        return false;
    }
    const unsigned char* data = reinterpret_cast<const unsigned char*>(pathStr.data());
    size_t len = pathStr.length();
    size_t i = 0;

    while (i < len) {
        unsigned char c = data[i];

        if (c < 0x80) {
            // 单字节：ASCII 字符
            if (!IsValidOutputPathChar(static_cast<char>(c))) {
                return false;
            }
            i++;
        } else if (c >= 0xE4 && c <= 0xE9 && i + 2 < len) {
            // 可能是中文：必须是以 0xE4~0xE9 开头的三字节 UTF-8
            if (IsChineseCharacter(data + i)) {
                i += 3;  // 跳过三个字节
            } else {
                return false;  // 不是合法汉字
            }
        } else {
            // 所有其他多字节情况都不允许：
            // - 0x80~0xC1: 非法起始字节
            // - 0xC0~0xDF: 双字节字符（如 é, ñ, α 等）→ 拒绝
            // - 0xF0~0xF7: 四字节字符（emoji、罕见字）→ 拒绝
            // - 0xEA~0xEF: 虽然也是三字节，但超出汉字范围（如其他 CJK 扩展区）→ 拒绝
            return false;
        }
    }
    return true;
}

bool IsValidDataLevel(const std::string &level)
{
    // 允许的值：0、1、op、kernel
    return (level == "0" || level == "1" || level == "op" || level == "kernel");
}

bool IsValidInteger(const std::string &str, const IntValidateRule &rule)
{
    // 1. 空串非法
    if (str.empty()) {
        return false;
    }

    // 2. 所有字符必须是数字
    for (char c : str) {
        if (c < '0' || c > '9') {
            return false;
        }
    }

    // 3. 规则：是否允许前导0（多位数时）
    if (!rule.allowLeadingZero && str.size() > 1 && str[0] == '0') {
        return false;
    }

    // 4. 规则：数值范围（minValue ≤ value ≤ maxValue）
    uint32_t value;
    if (!Utility::StrToUint32(value, str)) {
        return false;
    }
    return (value >= rule.minValue) && (value <= rule.maxValue);
}
}