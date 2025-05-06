// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include "config_info.h"
#include "file.h"

#define private public
#include "stepinter_analyzer.h"
#undef private

using namespace Leaks;

void CreateCsvData(CSV_FIELD_DATA &data)
{
    std::unordered_map<std::string, std::string> temp;
    temp["Event"] = "MALLOC";
    temp["Event Type"] = "PTA";
    temp["Attr"] = "{addr:1000,size:123,owner:,MID:3}";
    temp["Device Id"] = "0";
    temp["Timestamp(us)"] = "1";
    data.emplace_back(temp);
    temp["Event"] = "KERNEL_LAUNCH";
    temp["Event Type"] = "KERNEL_LAUNCH";
    temp["Name"] = "matmul_v1";
    temp["Timestamp(us)"] = "2";
    data.emplace_back(temp);
    temp["Event"] = "KERNEL_LAUNCH";
    temp["Event Type"] = "KERNEL_LAUNCH";
    temp["Name"] = "matmul_v2";
    temp["Timestamp(us)"] = "3";
    data.emplace_back(temp);
    temp["Event"] = "MALLOC";
    temp["Event Type"] = "PTA";
    temp["Attr"] = "{addr:1001,size:124,owner:,MID:3}";
    temp["Timestamp(us)"] = "4";
    data.emplace_back(temp);
}

TEST(StepInterAnalyzerTest, do_split_line_data_expect_split_string)
{
    Config config;
    std::string str = "testCase1,testCase2,testCase3";
    std::vector<std::string> expect = {"testCase1", "testCase2", "testCase3"};
    std::vector<std::string> result;
    StepInterAnalyzer stepinteranalyzer{config};
    result = stepinteranalyzer.SplitLineData(str);
    ASSERT_EQ(expect, result);
}

TEST(StepInterAnalyzerTest, do_read_csv_file_expect_read_correct_data)
{
    Utility::UmaskGuard umaskGuard(Utility::DEFAULT_UMASK_FOR_CSV_FILE);
    FILE *fp = fopen("test_leaks.csv", "w");
    std::string testHeader = std::string(LEAKS_HEADERS) + "\n";
    fprintf(fp, testHeader.c_str());

    for (int index = 0; index < 10; ++index) {
        fprintf(fp, "1,MALLOC,PTA,N/A,%d,123,234,0,0,N/A\n", index);
    }

    for (int index = 0; index < 5; ++index) {
        fprintf(fp, "2,KERNEL_LAUNCH,KERNEL_LAUNCH,null,%d,123,234,0,0,N/A\n", index+10);
    }

    for (int index = 0; index < 5; ++index) {
        fprintf(fp, "3,MALLOC,PTA,N/A,%d,123,234,2,0,N/A\n", index+2);
    }

    for (int index = 4; index >= 0; --index) {
        fprintf(fp, "4,KERNEL_LAUNCH,KERNEL_LAUNCH,null,%d,123,234,2,0,N/A\n", index+1);
    }

    fclose(fp);
    Config config;
    config.enableCStack = false;
    config.enablePyStack = false;
    StepInterAnalyzer stepinteranalyzer{config};
    std::unordered_map<DEVICEID, CSV_FIELD_DATA> data;
    std::string str = "test_leaks.csv";
    stepinteranalyzer.ReadCsvFile(str, data);
    ASSERT_EQ(data.size(), 2);
    ASSERT_EQ(data[0].size(), 15);
    ASSERT_EQ(data[2].size(), 10);
    remove("test_leaks.csv");
}

TEST(StepInterAnalyzerTest, do_read_invalid_csv_file_expect_empty_data)
{
    Utility::UmaskGuard umaskGuard(Utility::DEFAULT_UMASK_FOR_CSV_FILE);
    Config config;
    FILE *fp = fopen("test_leaks.csv", "w");
    std::string headers = "testheader1,testheader2\n";
    fprintf(fp, headers.c_str());

    fclose(fp);
    StepInterAnalyzer stepinteranalyzer{config};
    std::unordered_map<DEVICEID, CSV_FIELD_DATA> data;
    std::string str = "test_leaks.csv";
    stepinteranalyzer.ReadCsvFile(str, data);
    ASSERT_EQ(data.size(), 0);
    remove("test_leaks.csv");
    str = "test.csv";
    stepinteranalyzer.ReadCsvFile(str, data);
    ASSERT_EQ(data.size(), 0);

    fp = fopen("test_leaks.csv", "w");
    std::string testHeader = std::string(LEAKS_HEADERS) + "\n";
    fprintf(fp, testHeader.c_str());
    fprintf(fp, "1,0,pytorch,malloc,123,234,0,0,N/A,N/A,0\n");
    fclose(fp);
    stepinteranalyzer.ReadCsvFile(str, data);
    ASSERT_EQ(data.size(), 0);
    remove("test_leaks.csv");
}

TEST(StepInterAnalyzerTest, do_read_kernelLaunch_data_expect_return_true_and_correct_data)
{
    Config config;
    CSV_FIELD_DATA data;
    CreateCsvData(data);
    StepInterAnalyzer stepinteranalyzer{config};
    KERNELNAME_INDEX result;
    auto ret = stepinteranalyzer.ReadKernelLaunchData(data, result);
    ASSERT_TRUE(ret);
    ASSERT_EQ(result.size(), 2);
    ASSERT_EQ(result[0].first, "matmul_v1");
    ASSERT_EQ(result[0].second, 1);
    ASSERT_EQ(result[1].first, "matmul_v2");
    ASSERT_EQ(result[1].second, 2);
}

TEST(StepInterAnalyzerTest, do_read_no_kernelLaunch_data_expect_return_false_and_empty_data)
{
    Config config;
    CSV_FIELD_DATA data;
    StepInterAnalyzer stepinteranalyzer{config};
    KERNELNAME_INDEX result;
    auto ret = stepinteranalyzer.ReadKernelLaunchData(data, result);
    ASSERT_TRUE(ret);
    ASSERT_EQ(result.size(), 0);
}

TEST(StepInterAnalyzerTest, do_read_invalid_kernelLaunch_data_expect_falseand_empty_data)
{
    Config config;
    CSV_FIELD_DATA data;
    CreateCsvData(data);
    data[1]["Name"] = "+test";
    StepInterAnalyzer stepinteranalyzer{config};
    KERNELNAME_INDEX result;
    auto ret = stepinteranalyzer.ReadKernelLaunchData(data, result);
    ASSERT_FALSE(ret);
    ASSERT_EQ(result.size(), 0);
}

TEST(StepInterAnalyzerTest, do_get_kernel_memory_diff_expect_correct_data)
{
    Config config;
    CSV_FIELD_DATA data;
    CreateCsvData(data);
    StepInterAnalyzer stepinteranalyzer{config};
    int64_t result;
    stepinteranalyzer.GetKernelMemoryDiff(1, data, result);
    ASSERT_EQ(result, 124);

    stepinteranalyzer.GetKernelMemoryDiff(2, data, result);
    ASSERT_EQ(result, 124);

    data[3]["Attr"] = "test";
    stepinteranalyzer.GetKernelMemoryDiff(1, data, result);

    data[3]["Event Type"] = "KERNEL_LAUNCH";
    stepinteranalyzer.GetKernelMemoryDiff(1, data, result);
    ASSERT_EQ(result, 0);
}

TEST(StepInterAnalyzerTest, do_write_compare_data_to_csv_expect_true)
{
    Config config;
    StepInterAnalyzer stepinteranalyzer{config};
    std::string temp;
    temp = "matmul,0,10,0,11,20,0,21,10,0,10\n";
    stepinteranalyzer.compareOut_[0].emplace_back(temp);
    Utility::UmaskGuard umaskGuard(Utility::DEFAULT_UMASK_FOR_CSV_FILE);
    FILE *fp = fopen("test_leaks.csv", "w");
    stepinteranalyzer.compareFile_ = fp;
    ASSERT_TRUE(stepinteranalyzer.WriteCompareDataToCsv());
    fclose(fp);
    remove("test_leaks.csv");
}

TEST(StepInterAnalyzerTest, do_empty_compare_data_to_csv_expect_false)
{
    Config config;
    Utility::UmaskGuard umaskGuard(Utility::DEFAULT_UMASK_FOR_CSV_FILE);
    StepInterAnalyzer stepinteranalyzer{config};
    FILE *fp = fopen("test_leaks.csv", "w");
    stepinteranalyzer.compareFile_ = fp;
    ASSERT_FALSE(stepinteranalyzer.WriteCompareDataToCsv());
    fclose(fp);
    remove("test_leaks.csv");
}

TEST(StepInterAnalyzerTest, do_save_compare_kernel_memory_expect_correct_data)
{
    Config config;
    CSV_FIELD_DATA data;
    CreateCsvData(data);
    StepInterAnalyzer stepinteranalyzer{config};
    stepinteranalyzer.output_[0] = data;
    data[3]["Attr"] = "{addr:20623378087936,size:130,owner:,total:0,used:2822144}";
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

    Config config;
    StepInterAnalyzer stepinteranalyzer{config};
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
    Config config;
    std::shared_ptr<PathNode> pathNode;
    KERNELNAME_INDEX kernelIndexMap {};
    KERNELNAME_INDEX kernelIndexCompareMap {};
    StepInterAnalyzer stepinteranalyzer{config};
    stepinteranalyzer.buildDiff(pathNode, 0, kernelIndexMap, kernelIndexCompareMap);
    ASSERT_EQ(stepinteranalyzer.compareOut_.size(), 0);
}

TEST(StepInterAnalyzerTest, do_build_diff_expect_correct_data)
{
    std::shared_ptr<PathNode> pathNode1 = std::make_shared<DiffNode>(0, 0, nullptr);
    std::shared_ptr<PathNode> pathNode2 = std::make_shared<DiffNode>(1, 0, pathNode1);
    std::shared_ptr<PathNode> pathNode3 = std::make_shared<DiffNode>(1, 1, pathNode2);
    std::shared_ptr<PathNode> pathNode4 = std::make_shared<Snake>(2, 2, pathNode3);

    Config config;
    CSV_FIELD_DATA data;
    CreateCsvData(data);
    StepInterAnalyzer stepinteranalyzer{config};
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

TEST(StepInterAnalyzerTest, set_dir_path)
{
    Config config;
    Utility::SetDirPath("/MyPath", std::string(OUTPUT_PATH));
    StepInterAnalyzer::GetInstance(config).SetDirPath();
    EXPECT_EQ(StepInterAnalyzer::GetInstance(config).dirPath_, "/MyPath/" + std::string(COMPARE_FILE));
}

TEST(StepInterAnalyzerTest, do_myersdiff_input_kernelLaunch_data)
{
    Config config;
    KERNELNAME_INDEX kernelIndexMap {};
    KERNELNAME_INDEX kernelIndexCompareMap {};
    StepInterAnalyzer stepinteranalyzer{config};
    stepinteranalyzer.MyersDiff(0, kernelIndexMap, kernelIndexCompareMap);
    ASSERT_EQ(stepinteranalyzer.compareOut_[0].size(), 0);

    CSV_FIELD_DATA data;
    CreateCsvData(data);
    data[1]["name"] = "mul";
    kernelIndexMap.clear();
    kernelIndexMap.emplace_back(std::make_pair("matmul_v1", 1));
    kernelIndexMap.emplace_back(std::make_pair("matmul_v2", 2));
    kernelIndexCompareMap.clear();
    kernelIndexCompareMap.emplace_back(std::make_pair("mul", 1));
    kernelIndexCompareMap.emplace_back(std::make_pair("matmul_v2", 2));

    stepinteranalyzer.MyersDiff(0, kernelIndexMap, kernelIndexCompareMap);
    ASSERT_EQ(stepinteranalyzer.compareOut_[0].size(), 3);
}

TEST(StepInterAnalyzerTest, do_stepinter_compare_input_invalid_path_return_empty_data)
{
    Config config;
    std::vector<std::string> paths;
    paths.emplace_back("test_path1");
    paths.emplace_back("test_path2");

    StepInterAnalyzer stepinteranalyzer{config};
    stepinteranalyzer.StepInterCompare(paths);
    ASSERT_EQ(stepinteranalyzer.output_.size(), 0);
}

TEST(StepInterAnalyzerTest, do_stepinter_compare)
{
    Config config;
    config.enableCStack = false;
    config.enablePyStack = false;
    Utility::UmaskGuard umaskGuard(Utility::DEFAULT_UMASK_FOR_CSV_FILE);
    FILE *fp = fopen("test_leaks.csv", "w");
    std::string testHeader = std::string(LEAKS_HEADERS) + "\n";
    fprintf(fp, testHeader.c_str());

    for (int index = 0; index < 10; ++index) {
        fprintf(fp, "1,MALLOC,PTA,N/A,%d,123,234,0,0,N/A\n", index);
    }

    for (int index = 0; index < 5; ++index) {
        fprintf(fp, "2,KERNEL_LAUNCH,KERNEL_LAUNCH,null,%d,123,234,0,0,N/A\n", index+10);
    }

    for (int index = 0; index < 5; ++index) {
        fprintf(fp, "3,MALLOC,PTA,N/A,%d,123,234,2,0,N/A\n", index+2);
    }

    for (int index = 4; index >= 0; --index) {
        fprintf(fp, "4,KERNEL_LAUNCH,KERNEL_LAUNCH,null,%d,123,234,2,0,N/A\n", index+1);
    }

    fclose(fp);

    std::vector<std::string> paths;
    paths.emplace_back("test_leaks.csv");
    paths.emplace_back("test_leaks.csv");

    StepInterAnalyzer stepinteranalyzer{config};
    stepinteranalyzer.StepInterCompare(paths);
    ASSERT_EQ(stepinteranalyzer.output_.size(), 2);
    remove("test_leaks.csv");
}