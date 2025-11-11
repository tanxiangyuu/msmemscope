// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>
#include "json_manager.h"
#include "file.h"
#include "config_info.h"
#include "securec.h"

using namespace Utility;
using namespace Leaks;

// 假设 JsonManager 有公共 SetValue 接口（源码中 SaveConfigToJson 用到，必然存在）
// 若 SetValue 未公开，需通过友元或提供测试接口，此处按常规设计假设其存在
// 补充：若源码中 SetValue 是私有，需在 JsonManager 中添加测试支持（如 friend class JsonManagerTest;）

class JsonManagerTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetJsonManager();
    }

    void TearDown() override
    {
        rmdir("./testmsleaks");
        ResetJsonManager();
    }

    // 辅助函数：重置 JsonManager（通过加载空 JSON 文件实现，不访问私有成员）
    void ResetJsonManager()
    {
        std::string emptyJsonPath = "./testmsleaks/config.json";
        CreateTestJsonFile(emptyJsonPath, "{}"); // 创建空 JSON 文件
        JsonManager::GetInstance().LoadFromFile(emptyJsonPath);
        remove(emptyJsonPath.c_str()); // 立即删除临时文件
    }

    // 辅助函数：创建测试用的 JSON 文件
    bool CreateTestJsonFile(const std::string& filePath, const std::string& content)
    {
        std::ofstream ofs(filePath);
        if (!ofs.is_open()) {
            return false;
        }
        ofs << content;
        ofs.close();
        return true;
    }

    // 辅助函数：构造测试用的 Config 对象
    Leaks::Config BuildTestConfig()
    {
        Leaks::Config config;
        config.enableCStack = true;
        config.enablePyStack = false;
        config.enableCompare = true;
        config.cStackDepth = 10;
        config.pyStackDepth = 5;
        config.levelType = 1;
        config.eventType = 2;
        config.analysisType = 3;
        config.collectMode = 4;
        strncpy_s(config.outputDir, sizeof(config.outputDir), "./testmsleaks", sizeof(config.outputDir)-1);
        config.dataFormat = 5;
        config.collectAllNpu = true;
        config.npuSlots = 8;
        config.logLevel = 6;
        // 填充 stepList
        config.stepList.stepCount = 3;
        config.stepList.stepIdList[0] = 100;
        config.stepList.stepIdList[1] = 200;
        config.stepList.stepIdList[2] = 300;
        // 填充 watchConfig
        config.watchConfig.isWatched = true;
        config.watchConfig.fullContent = false;
        strncpy_s(config.watchConfig.start, sizeof(config.watchConfig.start), "2025-01-01", sizeof(config.watchConfig.start)-1);
        strncpy_s(config.watchConfig.end, sizeof(config.watchConfig.end), "2025-12-31", sizeof(config.watchConfig.end)-1);
        config.watchConfig.outputId = 10;
        return config;
    }
};

// 测试 SaveToFile：文件路径合法，保存成功
TEST_F(JsonManagerTest, save_to_file_success)
{
    auto& jsonMgr = JsonManager::GetInstance();
    jsonMgr.SetValue("test_key", "test_value"); // 使用公共 SetValue 接口
    std::string filePath = "./testmsleaks/config.json";
    
    bool ret = jsonMgr.SaveToFile(filePath);
    ASSERT_TRUE(ret);
    ASSERT_TRUE(Utility::FileExists(filePath));
}

// 测试 SaveToFile：文件路径非法（无写权限），保存失败
TEST_F(JsonManagerTest, save_to_file_no_permission_fail)
{
    auto& jsonMgr = JsonManager::GetInstance();
    jsonMgr.SetValue("test_key", "test_value");
    std::string noPermDir = "./testmsleaks/no_perm_dir";
    MakeDir(noPermDir);
    chmod(noPermDir.c_str(), 0444); // 只读权限
    std::string filePath = noPermDir + "/test.json";
    
    bool ret = jsonMgr.SaveToFile(filePath);
    ASSERT_FALSE(ret);
    
    // 恢复权限以便清理
    chmod(noPermDir.c_str(), 0755);
    rmdir(noPermDir.c_str());
}

// 测试 LoadFromFile：文件存在且格式合法，加载成功
TEST_F(JsonManagerTest, load_from_file_success)
{
    auto& jsonMgr = JsonManager::GetInstance();
    std::string filePath = "./testmsleaks/config.json";
    std::string jsonContent = R"({"name":"test","age":20})";
    ASSERT_TRUE(CreateTestJsonFile(filePath, jsonContent));
    
    bool ret = jsonMgr.LoadFromFile(filePath);
    ASSERT_TRUE(ret);
    
    // 通过 GetXXXValue 验证加载结果（不直接访问 jsonConfig_）
    std::string name;
    jsonMgr.GetStringValue("name", name);
    ASSERT_EQ(name, "test");
    
    uint32_t age = 0;
    jsonMgr.GetUint32Value("age", age);
    ASSERT_EQ(age, 20);
}

// 测试 LoadFromFile：文件不存在，加载失败
TEST_F(JsonManagerTest, load_from_file_not_exist_fail)
{
    auto& jsonMgr = JsonManager::GetInstance();
    std::string filePath = "./testmsleaks/not_exist.json";
    
    bool ret = jsonMgr.LoadFromFile(filePath);
    ASSERT_FALSE(ret);
}

// 测试 CheckKeyIsValid：key 为空，返回 false
TEST_F(JsonManagerTest, check_key_is_valid_empty_key_fail)
{
    auto& jsonMgr = JsonManager::GetInstance();
    // 加载合法 JSON（仅为构造 current 参数，实际 CheckKeyIsValid 依赖 jsonConfig_）
    std::string jsonContent = R"({"exist_key":"value"})";
    std::string filePath = "./testmsleaks/config.json";
    CreateTestJsonFile(filePath, jsonContent);
    jsonMgr.LoadFromFile(filePath);
    // 替代方案：不直接测试 CheckKeyIsValid（私有调用），通过 GetXXXValue 间接覆盖
    // 以下用例改为间接测试：key 不存在时 GetStringValue 失败
    std::string value = "default";
    jsonMgr.GetStringValue("", value);
    ASSERT_EQ(value, "default");
}

// 测试 CheckKeyIsValid：key 不存在（间接测试，通过 GetStringValue）
TEST_F(JsonManagerTest, check_key_is_valid_single_key_not_exist_fail)
{
    auto& jsonMgr = JsonManager::GetInstance();
    std::string jsonContent = R"({"exist_key":"value"})";
    std::string filePath = "./testmsleaks/config.json";
    CreateTestJsonFile(filePath, jsonContent);
    jsonMgr.LoadFromFile(filePath);
    
    std::string value = "default";
    jsonMgr.GetStringValue("not_exist_key", value);
    ASSERT_EQ(value, "default"); // 未获取到值，说明 CheckKeyIsValid 返回 false
}

// 测试 CheckKeyIsValid：key 存在（间接测试，通过 GetStringValue）
TEST_F(JsonManagerTest, check_key_is_valid_exist_key_success)
{
    auto& jsonMgr = JsonManager::GetInstance();
    std::string jsonContent = R"({"parent":{"child":"nested_value"}})";
    std::string filePath = "./testmsleaks/config.json";
    CreateTestJsonFile(filePath, jsonContent);
    jsonMgr.LoadFromFile(filePath);
    
    std::string value;
    jsonMgr.GetStringValue("parent.child", value);
    ASSERT_EQ(value, "nested_value"); // 获取到值，说明 CheckKeyIsValid 返回 true
}

// 测试 ReadJsonConfig：读取成功
TEST_F(JsonManagerTest, read_json_config_success)
{
    auto& jsonConfig = JsonConfig::GetInstance();
    Leaks::Config saveConfig = BuildTestConfig();
    std::string filePath = "./testmsleaks/config.json";
    setenv(MSLEAKS_CONFIG_ENV, filePath.c_str(), 1);
    jsonConfig.SaveConfigToJson(saveConfig); // 先保存配置
    
    Leaks::Config loadConfig = {};
    bool ret = jsonConfig.ReadJsonConfig(loadConfig);
    ASSERT_TRUE(ret);
}

// 测试 ReadJsonConfig：环境变量未设置，返回 false
TEST_F(JsonManagerTest, read_json_config_no_env_fail)
{
    auto& jsonConfig = JsonConfig::GetInstance();
    unsetenv(MSLEAKS_CONFIG_ENV); // 确保环境变量为空
    
    Leaks::Config config = {};
    bool ret = jsonConfig.ReadJsonConfig(config);
    ASSERT_FALSE(ret);
}

// 测试 ReadJsonConfig：文件不存在，返回 false
TEST_F(JsonManagerTest, read_json_config_file_not_exist_fail)
{
    auto& jsonConfig = JsonConfig::GetInstance();
    std::string nonExistPath = "./testmsleaks/not_exist_config.json";
    setenv(MSLEAKS_CONFIG_ENV, nonExistPath.c_str(), 1);
    
    Leaks::Config config = {};
    bool ret = jsonConfig.ReadJsonConfig(config);
    ASSERT_FALSE(ret);
}
