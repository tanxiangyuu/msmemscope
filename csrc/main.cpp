// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include <cstdint>
#include <iostream>
#include "framework/client_parser.h"
#include "utility/log.h"
int32_t main(int32_t argc, char **argv)
{
    Leaks::ClientParser parser;
    parser.Interpretor(argc, argv);
    return 0;
}