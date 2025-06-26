// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include <gtest/gtest.h>
#include "vallina_symbol.h"

using namespace Leaks;

TEST(VallinaSymbolTEST, lib_load_get_empty_expect_failed)
{
    std::string emptyLibName = "";
    EXPECT_EQ(Leaks::LibLoad(emptyLibName), nullptr);
}

TEST(VallinaSymbolTEST, join_path_get_empty_expect_get_component)
{
    std::string base = "";
    std::string component = "component";
    EXPECT_EQ(Leaks::JoinPath(base, component), component);
}

TEST(VallinaSymbolTEST, join_path_get_empty_expect_get_path)
{
    std::string base = "path/";
    std::string component = "component";
    EXPECT_EQ(Leaks::JoinPath(base, component), base + component);
}

TEST(VallinaSymbolTEST, find_parent_dir_get_empty_expect_failed)
{
    std::string startPath = "path/";
    int maxDepth = -1;
    EXPECT_EQ(Leaks::FindLibParentDir(startPath, maxDepth), "");
}

TEST(VallinaSymbolTEST, get_dir_name_get_empty)
{
    std::string startPath = "path";
    EXPECT_EQ(Leaks::GetDirname(startPath), ".");
}

TEST(VallinaSymbolTEST, find_and_load_sqlite_in_dir_get_wrong_depth_expect_failed)
{
    std::string startPath = "path";
    int depth = 1;
    int maxDepth = 0;
    EXPECT_EQ(Leaks::FindAndLoadSqliteInDir(startPath, depth, maxDepth), nullptr);
}