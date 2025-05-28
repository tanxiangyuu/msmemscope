// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef CALCULATE_MD5_H
#define CALCULATE_MD5_H

#include <string>
#include <vector>

// 定义 OpenSSL MD5 函数类型
using MD5Func = unsigned char* (*)(const unsigned char*, size_t, unsigned char*);
std::string GetTensorMD5(const std::vector<char>& data);
#endif