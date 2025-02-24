// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include "config_info.h"

#define private public
#include "stepinter_analyzer.h"
#undef private

using namespace Leaks;

void CreateCsvData(CSV_FIELD_DATA &data)
{
    std::unordered_map<std::string, std::string> temp;
    temp["Event"] = "pytorch";
    temp["Event Type"] = "malloc";
    temp["Size(byte)"] = "123";
    temp["Device Id"] = "0";
    temp["Timestamp(us)"] = "1";
    data.emplace_back(temp);
    temp["Event"] = "kernelLaunch";
    temp["Event Type"] = "matmul_v1";
    temp["Timestamp(us)"] = "2";
    data.emplace_back(temp);
    temp["Event"] = "kernelLaunch";
    temp["Event Type"] = "matmul_v2";
    temp["Timestamp(us)"] = "3";
    data.emplace_back(temp);
    temp["Event"] = "pytorch";
    temp["Event Type"] = "malloc";
    temp["Size(byte)"] = "124";
    temp["Timestamp(us)"] = "4";
    data.emplace_back(temp);
}

TEST(StepInterAnalyzerTest, do_split_line_data_expect_split_string)
{
    std::string str = "testCase1,testCase2,testCase3";
    std::vector<std::string> expect = {"testCase1", "testCase2", "testCase3"};
    std::vector<std::string> result;
    StepInterAnalyzer stepinteranalyzer{};
    result = stepinteranalyzer.SplitLineData(str);
    ASSERT_EQ(expect, result);
}

TEST(StepInterAnalyzerTest, do_read_csv_file_expect_read_correct_data)
{
    FILE *fp = fopen("test_leaks.csv", "w");
    fprintf(fp, LEAKS_HEADERS);

    for (int index = 0; index < 10; ++index) {
        fprintf(fp, "1,%d,pytorch,malloc,123,234,0,0,N/A,N/A,0,%d,%d\n", index, index+100, index+1000);
    }

    for (int index = 0; index < 5; ++index) {
        fprintf(fp, "2,%d,kernelLaunch,null,123,234,0,0,N/A,N/A,0,0,0\n", index+10);
    }

    for (int index = 0; index < 5; ++index) {
        fprintf(fp, "3,%d,pytorch,malloc,123,234,2,0,N/A,N/A,0,%d,%d\n", index+2, index+100, index+1000);
    }

    for (int index = 4; index >= 0; --index) {
        fprintf(fp, "4,%d,kernelLaunch,null,123,234,2,0,N/A,N/A,0,0,0\n", index+1);
    }

    fclose(fp);
    StepInterAnalyzer stepinteranalyzer{};
    std::unordered_map<DEVICEID, CSV_FIELD_DATA> data;
    stepinteranalyzer.ReadCsvFile("test_leaks.csv", data);
    ASSERT_EQ(data.size(), 2);
    ASSERT_EQ(data[0].size(), 15);
    ASSERT_EQ(data[2].size(), 10);
    remove("test_leaks.csv");
}

TEST(StepInterAnalyzerTest, do_read_invalid_csv_file_expect_empty_data)
{
    FILE *fp = fopen("test_leaks.csv", "w");
    std::string headers = "testheader1,testheader2\n";
    fprintf(fp, headers.c_str());

    fclose(fp);
    StepInterAnalyzer stepinteranalyzer{};
    std::unordered_map<DEVICEID, CSV_FIELD_DATA> data;
    stepinteranalyzer.ReadCsvFile("test_leaks.csv", data);
    ASSERT_EQ(data.size(), 0);
    remove("test_leaks.csv");

    stepinteranalyzer.ReadCsvFile("test.csv", data);
    ASSERT_EQ(data.size(), 0);
}

TEST(StepInterAnalyzerTest, do_read_kernelLaunch_data_expect_cprrect_data)
{
    CSV_FIELD_DATA data;
    CreateCsvData(data);
    StepInterAnalyzer stepinteranalyzer{};
    KERNELNAME_INDEX result = stepinteranalyzer.ReadKernelLaunchData(data);
    ASSERT_EQ(result.size(), 2);
    ASSERT_EQ(result[0].first, "matmul_v1");
    ASSERT_EQ(result[0].second, 1);
    ASSERT_EQ(result[1].first, "matmul_v2");
    ASSERT_EQ(result[1].second, 2);
}

TEST(StepInterAnalyzerTest, do_read_no_kernelLaunch_data_expect_empty_data)
{
    CSV_FIELD_DATA data;
    StepInterAnalyzer stepinteranalyzer{};
    KERNELNAME_INDEX result = stepinteranalyzer.ReadKernelLaunchData(data);
    ASSERT_EQ(result.size(), 0);
}

TEST(StepInterAnalyzerTest, do_get_kernel_memory_diff_expect_correct_data)
{
    CSV_FIELD_DATA data;
    CreateCsvData(data);
    StepInterAnalyzer stepinteranalyzer{};
    int64_t result;
    stepinteranalyzer.GetKernelMemoryDiff(1, data, result);
    ASSERT_EQ(result, 124);

    stepinteranalyzer.GetKernelMemoryDiff(2, data, result);
    ASSERT_EQ(result, 124);
}

TEST(StepInterAnalyzerTest, do_write_compare_data_to_csv_expect_true)
{
    StepInterAnalyzer stepinteranalyzer{};
    std::string temp;
    temp = "matmul,0,10,0,11,20,0,21,10,0,10\n";
    stepinteranalyzer.compareOut_[0].emplace_back(temp);
    FILE *fp = fopen("test_leaks.csv", "w");
    stepinteranalyzer.compareFile_ = fp;
    ASSERT_TRUE(stepinteranalyzer.WriteCompareDataToCsv());
    fclose(fp);
    remove("test_leaks.csv");
}

TEST(StepInterAnalyzerTest, do_empty_compare_data_to_csv_expect_false)
{
    StepInterAnalyzer stepinteranalyzer{};
    FILE *fp = fopen("test_leaks.csv", "w");
    stepinteranalyzer.compareFile_ = fp;
    ASSERT_FALSE(stepinteranalyzer.WriteCompareDataToCsv());
    fclose(fp);
    remove("test_leaks.csv");
}

TEST(StepInterAnalyzerTest, do_save_compare_kernel_memory_expect_correct_data)
{
    CSV_FIELD_DATA data;
    CreateCsvData(data);
    StepInterAnalyzer stepinteranalyzer{};
    stepinteranalyzer.output_[0] = data;
    data[3]["Size(byte)"] = "130";
    stepinteranalyzer.outputCompare_[0] = data;

    std::string temp;
    temp = "matmul_v1,0,124,130,6";

    stepinteranalyzer.SaveCompareKernelMemory(0, {"matmul_v1", 1}, {"matmul_v1", 1});
    ASSERT_EQ(stepinteranalyzer.compareOut_[0][0], temp);

    stepinteranalyzer.SaveCompareKernelMemory(0, {"matmul_v2", 2}, {"matmul_v2", 2});
    temp = "matmul_v2,0,124,130,6";
    ASSERT_EQ(stepinteranalyzer.compareOut_[0][1], temp);
}

TEST(StepInterAnalyzerTest, do_build_path_expect_coorrect_data)
{
    KERNELNAME_INDEX kernelIndexMap {};
    kernelIndexMap.emplace_back(std::make_pair("matmul_v1", 1));
    kernelIndexMap.emplace_back(std::make_pair("matmul_v2", 2));
    KERNELNAME_INDEX kernelIndexCompareMap {};
    kernelIndexCompareMap.emplace_back(std::make_pair("mul", 1));
    kernelIndexCompareMap.emplace_back(std::make_pair("matmul_v2", 2));

    std::shared_ptr<PathNode> pathNode1 = std::make_shared<DiffNode>(0, 0, nullptr);
    std::shared_ptr<PathNode> pathNode2 = std::make_shared<DiffNode>(1, 0, pathNode1);
    std::shared_ptr<PathNode> pathNode3 = std::make_shared<DiffNode>(1, 1, pathNode2);
    std::shared_ptr<PathNode> pathNode4 = std::make_shared<Snake>(2, 2, pathNode3);

    StepInterAnalyzer stepinteranalyzer{};
    auto pathNode = stepinteranalyzer.buildPath(kernelIndexMap, kernelIndexCompareMap);
    ASSERT_EQ(pathNode->i, pathNode4->i);
    ASSERT_EQ(pathNode->j, pathNode4->j);
    pathNode = pathNode->prev;
    ASSERT_EQ(pathNode->i, pathNode3->i);
    ASSERT_EQ(pathNode->j, pathNode3->j);
    pathNode = pathNode->prev;
    ASSERT_EQ(pathNode->i, pathNode2->i);
    ASSERT_EQ(pathNode->j, pathNode2->j);
    pathNode = pathNode->prev;
    ASSERT_EQ(pathNode->i, pathNode1->i);
    ASSERT_EQ(pathNode->j, pathNode1->j);
}

TEST(StepInterAnalyzerTest, do_empty_path_build_diff_expect_empty_data)
{
    std::shared_ptr<PathNode> pathNode;
    KERNELNAME_INDEX kernelIndexMap {};
    KERNELNAME_INDEX kernelIndexCompareMap {};
    StepInterAnalyzer stepinteranalyzer{};
    stepinteranalyzer.buildDiff(pathNode, 0, kernelIndexMap, kernelIndexCompareMap);
    ASSERT_EQ(stepinteranalyzer.compareOut_.size(), 0);
}

TEST(StepInterAnalyzerTest, do_build_diff_expect_correct_data)
{
    std::shared_ptr<PathNode> pathNode1 = std::make_shared<DiffNode>(0, 0, nullptr);
    std::shared_ptr<PathNode> pathNode2 = std::make_shared<DiffNode>(1, 0, pathNode1);
    std::shared_ptr<PathNode> pathNode3 = std::make_shared<DiffNode>(1, 1, pathNode2);
    std::shared_ptr<PathNode> pathNode4 = std::make_shared<Snake>(2, 2, pathNode3);

    CSV_FIELD_DATA data;
    CreateCsvData(data);
    StepInterAnalyzer stepinteranalyzer{};
    stepinteranalyzer.output_[0] = data;
    data[1]["name"] = "mul";
    stepinteranalyzer.outputCompare_[0] = data;
    KERNELNAME_INDEX kernelIndexMap {};
    kernelIndexMap.emplace_back(std::make_pair("matmul_v1", 1));
    kernelIndexMap.emplace_back(std::make_pair("matmul_v2", 2));
    KERNELNAME_INDEX kernelIndexCompareMap {};
    kernelIndexCompareMap.emplace_back(std::make_pair("mul", 1));
    kernelIndexCompareMap.emplace_back(std::make_pair("matmul_v2", 2));

    stepinteranalyzer.buildDiff(pathNode4, 0, kernelIndexMap, kernelIndexCompareMap);
    ASSERT_EQ(stepinteranalyzer.compareOut_[0].size(), 3);
}
