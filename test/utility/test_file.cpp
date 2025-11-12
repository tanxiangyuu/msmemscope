// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include <gtest/gtest.h>
#include <dlfcn.h>
#define private public
#include "utility/file.h"
#undef private
#include "utility/log.h"
#include "config_info.h"

using namespace Utility;

class FileTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        Utility::FileCreateManager::GetInstance("./testmsleaks");
    }
 
    void TearDown() override
    {
        Utility::FileCreateManager::GetInstance("./testmsleaks").SetProjectDir("");
        rmdir("./testmsleaks");
    }
};

TEST_F(FileTest, make_empty_path_expect_return_false)
{
    std::string dirPath;
    auto ret = MakeDir(dirPath);
    ASSERT_FALSE(ret);
}

TEST_F(FileTest, make_path_expect_return_true)
{
    std::string dirPath = "test_dir";
    auto ret = MakeDir(dirPath);
    ASSERT_TRUE(ret);
    rmdir(dirPath.c_str());
}

TEST_F(FileTest, make_recursive_path_expect_return_true)
{
    std::string dirPath = "./test_dir/test1/test2";
    auto ret = MakeDir(dirPath);
    ASSERT_TRUE(ret);
    rmdir(dirPath.c_str());
    rmdir("./test_dir/test1");
    rmdir("./test_dir");
}


TEST_F(FileTest, make_exist_path_expect_return_false)
{
    char cwd[100];
    getcwd(cwd, sizeof(cwd));
    std::string dirPath = cwd;
    auto ret = MakeDir(dirPath);
    ASSERT_TRUE(ret);
}

TEST_F(FileTest, not_exist_path_expect_return_false)
{
    std::string path;
    auto ret = Exist(path);
    ASSERT_FALSE(ret);
}

TEST_F(FileTest, exist_path_expect_return_false)
{
    std::string path = __FILE__;
    auto ret = Exist(path);
    ASSERT_TRUE(ret);
}

TEST_F(FileTest, create_new_csv_file_expect_true)
{
    FILE *fp = nullptr;
    FileCreateManager::GetInstance("").projectDir_ = "./testmsleaks";
    auto ret = FileCreateManager::GetInstance("./testmsleaks").CreateCsvFile(&fp, "0", "test",
        Leaks::DUMP_DIR, "test_headers\n");
    ASSERT_TRUE(ret);
    fclose(fp);
    remove("./testmsleaks/test.csv");
    rmdir("./testmsleaks");
}

TEST_F(FileTest, create_file_with_umask_failed)
{
    uint32_t mask = 0177;
    std::string path = "";
    FileCreateManager::GetInstance("").projectDir_ = path;
    auto ret = FileCreateManager::GetInstance(path).CreateFileWithUmask(path, "", mask);
    EXPECT_EQ(ret, nullptr);
}