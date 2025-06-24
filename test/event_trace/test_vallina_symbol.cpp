// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include <gtest/gtest.h>
#include "vallina_symbol.h"

using namespace Leaks;

TEST(VallinaSymbolTEST, lib_load_from_drvier_get_empty_expect_failed)
{
    std::string emptyLibName = "";
    EXPECT_EQ(Leaks::LibLoadFromDriver(emptyLibName), nullptr);
}
