// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include <sstream>
#include <iomanip>
#include "calculate_md5.h"
#include "vallina_symbol.h"
#include "client_process.h"

namespace Utility {

std::string GetTensorMD5(const std::vector<char>& data)
{
    static auto func = Leaks::VallinaSymbol<Leaks::OpenSSLLibLoader>::Instance().Get<Utility::MD5Func>("MD5");
    if (func == nullptr) {
        CLIENT_ERROR_LOG("Cannot find MD5Func");
        return "";
    }

    // 计算哈希
    static size_t valueSize = 16;
    unsigned char digest[valueSize]; // MD5 结果为 16 字节
    func(reinterpret_cast<const unsigned char*>(data.data()), data.size(), digest);

    // 转换为十六进制字符串
    std::ostringstream oss;
    static size_t valueWide = 2;
    oss << std::hex << std::setfill('0');
    for (int i = 0; i < valueSize; i++) {
        oss << std::setw(valueWide) << static_cast<int>(digest[i]);
    }

    return oss.str();
}

}