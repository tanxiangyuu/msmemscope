// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include <gtest/gtest.h>

#include "utility/file.h"
#include "config_info.h"

using namespace Utility;

TEST(File, input_empty_path_expect_set_default_path)
{
    std::string pathStr;
    SetDirPath(pathStr, std::string(Leaks::OUTPUT_PATH));
    Utility::Path path = Utility::Path{std::string(Leaks::OUTPUT_PATH)};
    auto realPath = path.Resolved().ToString();
    ASSERT_EQ(g_dirPath, realPath);
    g_dirPath.clear();
}

TEST(File, input_path_expect_set_true_path)
{
    std::string pathStr = "test path";
    SetDirPath(pathStr, std::string(Leaks::OUTPUT_PATH));
    ASSERT_EQ(g_dirPath, pathStr);
    g_dirPath.clear();
}

TEST(File, make_empty_path_expect_return_false)
{
    std::string dirPath;
    auto ret = MakeDir(dirPath);
    ASSERT_FALSE(ret);
}

TEST(File, make_path_expect_return_true)
{
    std::string dirPath = "test_dir";
    auto ret = MakeDir(dirPath);
    ASSERT_TRUE(ret);
    rmdir(dirPath.c_str());
}

TEST(File, make_recursive_path_expect_return_true)
{
    std::string dirPath = "./test_dir/test1/test2";
    auto ret = MakeDir(dirPath);
    ASSERT_TRUE(ret);
    rmdir(dirPath.c_str());
    rmdir("./test_dir/test1");
    rmdir("./test_dir");
}


TEST(File, make_exist_path_expect_return_false)
{
    char cwd[100];
    getcwd(cwd, sizeof(cwd));
    std::string dirPath = cwd;
    auto ret = MakeDir(dirPath);
    ASSERT_TRUE(ret);
}

TEST(File, not_exist_path_expect_return_false)
{
    std::string path;
    auto ret = Exist(path);
    ASSERT_FALSE(ret);
}

TEST(File, exist_path_expect_return_false)
{
    std::string path = __FILE__;
    auto ret = Exist(path);
    ASSERT_TRUE(ret);
}

TEST(FILE, create_new_csv_file_expect_true)
{
    FILE *fp = nullptr;
    auto ret = CreateCsvFile(&fp, "./testmsleaks", "test", "test_headers\n");
    ASSERT_TRUE(ret);
    fclose(fp);
    remove("./testmsleaks/test.csv");
    rmdir("./testmsleaks");
}

TEST(FILE, create_empty_csv_file_expect_false)
{
    FILE *fp = nullptr;
    std::string dirPath;
    auto ret = CreateCsvFile(&fp, dirPath, "test.csv", "test_headers\n");
    ASSERT_FALSE(ret);
}
