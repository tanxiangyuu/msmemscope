// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "ustring.h"
#include <unordered_map>

namespace Utility {

void ToSafeString(std::string &str)
{
    static std::unordered_map<char, std::string> invalidChar = {{'\n', "\\n"},
        {'\f', "\\f"},
        {'\r', "\\r"},
        {'\b', "\\b"},
        {'\t', "\\t"},
        {'\v', "\\v"},
        {'\u007F', "\\u007F"}};
    for (size_t i = 0; i < str.size();) {
        if (invalidChar.find(str[i]) != invalidChar.end()) {
            std::string validStr = invalidChar[str[i]];
            str.replace(i, 1, validStr);
            i += validStr.length();
            continue;
        }
        i++;
    }
    return;
}

std::string ExtractAttrValueByKey(const std::string& str, const std::string& key)
{
    std::string attrValue = "";
    size_t startPos = str.find(key + ":");
    if (startPos != std::string::npos) {
        // 跳过键和冒号
        startPos += key.length() + 1;
        size_t endPos = str.find(",", startPos);
        if (endPos == std::string::npos) {
            endPos = str.find("}", startPos);
        }
        if (endPos != std::string::npos) {
            attrValue = str.substr(startPos, endPos - startPos);
        }
    }
    return attrValue;
}

} // namespace Utility