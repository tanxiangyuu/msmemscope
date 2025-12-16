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

#include <gtest/gtest.h>
#include "vallina_symbol.h"

using namespace MemScope;

TEST(VallinaSymbolTEST, lib_load_get_empty_expect_failed)
{
    std::string emptyLibName = "";
    EXPECT_EQ(MemScope::LibLoad(emptyLibName), nullptr);
}

TEST(VallinaSymbolTEST, join_path_get_empty_expect_get_component)
{
    std::string base = "";
    std::string component = "component";
    EXPECT_EQ(MemScope::JoinPath(base, component), component);
}

TEST(VallinaSymbolTEST, join_path_get_empty_expect_get_path)
{
    std::string base = "path/";
    std::string component = "component";
    EXPECT_EQ(MemScope::JoinPath(base, component), base + component);
}

TEST(VallinaSymbolTEST, find_parent_dir_get_empty_expect_failed)
{
    std::string startPath = "path/";
    int maxDepth = -1;
    EXPECT_EQ(MemScope::FindLibParentDir(startPath, maxDepth), "");
}

TEST(VallinaSymbolTEST, get_dir_name_get_empty)
{
    std::string startPath = "path";
    EXPECT_EQ(MemScope::GetDirname(startPath), ".");
}

TEST(VallinaSymbolTEST, find_and_load_sqlite_in_dir_get_wrong_depth_expect_failed)
{
    std::string startPath = "path";
    int depth = 1;
    int maxDepth = 0;
    EXPECT_EQ(MemScope::FindAndLoadSqliteInDir(startPath, depth, maxDepth), nullptr);
}