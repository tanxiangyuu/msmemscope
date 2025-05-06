// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "ustring.h"

namespace Utility {

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