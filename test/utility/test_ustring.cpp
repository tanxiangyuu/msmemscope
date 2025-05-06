// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "utility/ustring.h"

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <iterator>

using namespace Utility;

TEST(UString, join_empty_list_expect_empty_string)
{
    std::vector<std::string> items;
    std::string joined = Join(items.cbegin(), items.cend());
    ASSERT_TRUE(joined.empty());
}

TEST(UString, join_non_empty_list_expect_correct_string)
{
    std::vector<std::string> items = {
        "aaa",
        "bbb",
        "ccc"
    };
    std::string joined = Join(items.cbegin(), items.cend(), "::");
    ASSERT_EQ(joined, "aaa::bbb::ccc");
}

TEST(UString, strip_empty_string_expect_equal_to_self)
{
    std::string str = "";
    ASSERT_EQ(Strip(str), str);
}

TEST(UString, strip_string_with_target_expect_correct_string)
{
    std::string str = "::abc::";
    ASSERT_EQ(Strip(str, ":"), "abc");
}

TEST(UString, strip_string_without_target_expect_equal_to_self)
{
    std::string str = "abc::def";
    ASSERT_EQ(Strip(str, ":"), str);
}

TEST(UString, strip_string_twice_expect_equal_to_strip_once)
{
    std::string str = "::abc::";
    ASSERT_EQ(Strip(str, ":"), Strip(Strip(str, ":"), ":"));
}

TEST(UString, split_empty_string_expect_list_of_one_empty_string)
{
    std::vector<std::string> items;
    Split("", std::back_inserter(items));
    ASSERT_EQ(items.size(), 1UL);
    ASSERT_EQ(items[0], "");
}

TEST(UString, split_string_without_delims_expect_list_of_one_string)
{
    std::vector<std::string> items;
    Split("aaa", std::back_inserter(items));
    ASSERT_EQ(items.size(), 1UL);
    ASSERT_EQ(items[0], "aaa");
}

TEST(UString, split_string_start_with_delims_expect_list_start_with_empty_string)
{
    std::vector<std::string> items;
    Split(":aaa", std::back_inserter(items), ":");
    ASSERT_EQ(items.size(), 2UL);
    ASSERT_EQ(items[0], "");
    ASSERT_EQ(items[1], "aaa");
}

TEST(UString, split_string_end_with_delims_expect_list_end_with_empty_string)
{
    std::vector<std::string> items;
    Split("aaa:", std::back_inserter(items), ":");
    ASSERT_EQ(items.size(), 2UL);
    ASSERT_EQ(items[0], "aaa");
    ASSERT_EQ(items[1], "");
}

TEST(UString, split_string_with_muti_delims_test_strict_except_correct_list)
{
    std::vector<std::string> items;
    Split("aaa::bbb", std::back_inserter(items), ":", true);
    ASSERT_EQ(items.size(), 3UL);
    ASSERT_EQ(items[0], "aaa");
    ASSERT_EQ(items[1], "");
    ASSERT_EQ(items[2], "bbb");
}

TEST(UString, split_string_with_several_delims_expect_correct_list)
{
    std::vector<std::string> items;
    Split(":aaa:bbb:ccc:", std::back_inserter(items), ":");
    ASSERT_EQ(items.size(), 5UL);
    ASSERT_EQ(items[0], "");
    ASSERT_EQ(items[1], "aaa");
    ASSERT_EQ(items[2], "bbb");
    ASSERT_EQ(items[3], "ccc");
    ASSERT_EQ(items[4], "");
}

TEST(UString, extract_attr_value_by_key_expect_correct_value)
{
    std::string attrKey = "size";
    std::string str = "{addr:1000,size:123,owner:,MID:3}";
    std::string attrValue = ExtractAttrValueByKey(str, attrKey);
    ASSERT_EQ(attrValue, "123");
}

TEST(UString, extract_attr_value_by_key_expect_empty_value)
{
    std::string attrKey = "size";
    std::string str = "test";
    std::string attrValue = ExtractAttrValueByKey(str, attrKey);
    ASSERT_EQ(attrValue, "");
}
