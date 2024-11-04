// Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

#include <string>
#include <vector>
#include <iterator>
#include <unistd.h>
#include <gtest/gtest.h>
#include <linux/limits.h>

#include "utility/path.h"

using namespace Utility;

TEST(Path, default_construction_expect_current_path)
{
    Path path;
    ASSERT_EQ(path.ToString(), ".");
}

TEST(Path, parse_empty_path_expect_current_path)
{
    Path path("");
    ASSERT_EQ(path.ToString(), ".");
}

TEST(Path, parse_root_path_expect_get_root_path)
{
    Path path("/");
    ASSERT_EQ(path.ToString(), "/");
}

TEST(Path, parse_absolute_dir_path_expect_get_same_dir)
{
    std::string raw = "/usr/local";
    Path path(raw);
    ASSERT_EQ(path.ToString(), raw);
}

TEST(Path, parse_absolute_file_path_expect_get_same_file)
{
    std::string raw = "/usr/local/test.cpp";
    Path path(raw);
    ASSERT_EQ(path.ToString(), raw);
}

TEST(Path, parse_relative_dir_path_expect_get_correct_path)
{
    Path path("./test/");
    ASSERT_EQ(path.ToString(), "test");
}

TEST(Path, parse_relative_file_path_expect_get_correct_path)
{
    Path path("./test/test.cpp");
    ASSERT_EQ(path.ToString(), "test/test.cpp");
}

TEST(Path, parse_relative_dir_path_without_current_operator_expect_get_correct_path)
{
    Path path("test/");
    ASSERT_EQ(path.ToString(), "test");
}

TEST(Path, parse_relative_file_path_without_current_operator_expect_get_correct_path)
{
    Path path("test/test.cpp");
    ASSERT_EQ(path.ToString(), "test/test.cpp");
}

TEST(Path, parse_path_with_current_operator_expect_current_path)
{
    Path path1("test/./test.cpp");
    ASSERT_EQ(path1.ToString(), "test/test.cpp");
    Path path2("/test/./test.cpp");
    ASSERT_EQ(path2.ToString(), "/test/test.cpp");
}

TEST(Path, parse_path_with_parrent_operator_expect_correct_path)
{
    Path path1("test/../test.cpp");
    ASSERT_EQ(path1.ToString(), "test/../test.cpp");
    Path path2("/test/../test.cpp");
    ASSERT_EQ(path2.ToString(), "/test/../test.cpp");
}

TEST(Path, parse_path_with_multiple_seps_expect_ignore_redundant_seps)
{
    Path path("///test///test.cpp");
    ASSERT_EQ(path.ToString(), "/test/test.cpp");
}

TEST(Path, get_basename_of_absolute_dir_path_expect_return_correct_basename)
{
    Path path("/usr/local/");
    ASSERT_EQ(path.Name(), "local");
}

TEST(Path, get_basename_of_relative_dir_path_expect_return_correct_basename)
{
    Path path("local/");
    ASSERT_EQ(path.Name(), "local");
}

TEST(Path, get_basename_of_absolute_file_path_expect_return_correct_basename)
{
    Path path("/usr/local/test.cpp");
    ASSERT_EQ(path.Name(), "test.cpp");
}

TEST(Path, get_basename_of_relative_file_path_expect_return_correct_basename)
{
    Path path("./sample-dir/test.cpp");
    ASSERT_EQ(path.Name(), "test.cpp");
}

TEST(Path, get_basename_of_root_dir_expect_return_empty_basename)
{
    Path path("/");
    ASSERT_EQ(path.Name(), "");
}

TEST(Path, get_basename_of_current_dir_expect_return_empty_basename)
{
    Path path(".");
    ASSERT_EQ(path.Name(), "");
}

TEST(Path, get_parent_of_relative_path_expect_return_correct_parent)
{
    Path path("./sample-dir/test/test.cpp");
    path = path.Parent();
    ASSERT_EQ(path.ToString(), "sample-dir/test");
    path = path.Parent();
    ASSERT_EQ(path.ToString(), "sample-dir");
    path = path.Parent();
    ASSERT_EQ(path.ToString(), ".");
}

TEST(Path, get_parent_of_absolute_path_expect_return_correct_parent)
{
    Path path("/usr/local/test.cpp");
    path = path.Parent();
    ASSERT_EQ(path.ToString(), "/usr/local");
    path = path.Parent();
    ASSERT_EQ(path.ToString(), "/usr");
    path = path.Parent();
    ASSERT_EQ(path.ToString(), "/");
}

TEST(Path, get_parent_of_current_path_expect_return_current_parent)
{
    Path path(".");
    ASSERT_EQ(path.Parent().ToString(), path.ToString());
}

TEST(Path, get_parent_of_root_path_expect_return_root_parent)
{
    Path path("/");
    ASSERT_EQ(path.Parent().ToString(), path.ToString());
}

TEST(Path, get_absolute_path_of_absolute_path_expect_return_original_path)
{
    Path path("/usr/local");
    ASSERT_EQ(path.Absolute().ToString(), path.ToString());
}

TEST(Path, get_absolute_path_of_relative_path_expect_return_correct_path)
{
    Path path("./sample-dir/test/test.cpp");
    char buf[PATH_MAX] = {0};
    if (getcwd(buf, sizeof(buf))) {
        std::string cwd;
        cwd = buf;
        ASSERT_EQ(path.Absolute().ToString(), cwd + "/sample-dir/test/test.cpp");
    }
}

TEST(Path, get_absolute_path_expect_idempotence)
{
    Path path("/usr/local");
    ASSERT_EQ(path.Absolute().ToString(), path.Absolute().Absolute().ToString());
    path = Path("./sample-dir/test/test.cpp");
    ASSERT_EQ(path.Absolute().ToString(), path.Absolute().Absolute().ToString());
}

TEST(Path, get_resolved_path_of_absolute_path_expect_return_correct_path)
{
    Path path("/usr/local");
    ASSERT_EQ(path.Resolved().ToString(), path.ToString());
    path = Path("/usr/../local");
    ASSERT_EQ(path.Resolved().ToString(), "/local");
    path = Path("/usr/../../local");
    ASSERT_EQ(path.Resolved().ToString(), "/local");
}

TEST(Path, get_resolved_path_of_relative_path_expect_return_correct_path)
{
    char buf[PATH_MAX] = {0};
    if (getcwd(buf, sizeof(buf))) {
        std::string cwd;
        cwd = buf;
        Path path("./sample-dir/test.cpp");
        ASSERT_EQ(path.Resolved().ToString(), cwd + "/sample-dir/test.cpp");
        path = Path("./sample-dir/../test.cpp");
        ASSERT_EQ(path.Resolved().ToString(), cwd + "/test.cpp");
        path = Path("./sample-dir/../../test.cpp");
        ASSERT_EQ(path.Resolved().ToString(), Path(cwd).Parent().ToString() + "/test.cpp");
    }
}

TEST(Path, get_resolved_path_expect_idempotence)
{
    Path path("/usr/local");
    ASSERT_EQ(path.Resolved().ToString(), path.Resolved().Resolved().ToString());
    path = Path("./sample-dir/test/test.cpp");
    ASSERT_EQ(path.Resolved().ToString(), path.Resolved().Resolved().ToString());
}

TEST(Path, join_two_empty_pathes_expect_return_current_path_segment)
{
    Path path("");
    Path joined = path / path;
    ASSERT_EQ(joined.ToString(), ".");
}

TEST(Path, join_empty_path_with_another_path_expect_return_another_path)
{
    Path path1("");
    Path path2("/usr/local");
    Path path3("./test");
    Path joined = path1 / path2;
    ASSERT_EQ(joined.ToString(), path2.ToString());
    joined = path2 / path1;
    ASSERT_EQ(joined.ToString(), path2.ToString());
    joined = path1 / path3;
    ASSERT_EQ(joined.ToString(), path3.ToString());
    joined = path3 / path1;
    ASSERT_EQ(joined.ToString(), path3.ToString());
}

TEST(Path, join_several_pathed_expect_return_correct_path)
{
    Path path1("/usr/local");
    Path path2("test");
    Path path3("./test.cpp");
    Path joined = path1 / path2 / path3;
    ASSERT_EQ(joined.ToString(), "/usr/local/test/test.cpp");
}

TEST(Path, join_with_absolute_path_expect_return_only_absolute_path)
{
    Path path1("./test");
    Path path2("/usr/local");
    Path joined = path1 / path2;
    ASSERT_EQ(joined.ToString(), path2.ToString());
}

TEST(Path, check_exist_file_exists_expect_return_true)
{
    Path path("/bin/ls");
    ASSERT_TRUE(path.Exists());
}

TEST(Path, check_exist_dir_exists_expect_return_true)
{
    Path path("/bin/");
    ASSERT_TRUE(path.Exists());
}

TEST(Path, check_not_exist_path_exists_expect_return_false)
{
    Path path("./not_exist");
    ASSERT_FALSE(path.Exists());
}
