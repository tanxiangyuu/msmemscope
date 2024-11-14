// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include <gtest/gtest.h>
#include <unordered_map>
#include "kernel_hooks/runtime_hooks.h"
#include "kernel_hooks/acl_hooks.h"
#include "vallina_symbol.h"

using namespace testing;

RTS_API rtError_t MockRtKernelLaunch(
    const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize, rtSmDesc_t *smDesc, rtStream_t stm)
{
    std::cout << "Stub Func: " << __func__ << std::endl;
    return RT_ERROR_NONE;
}

RTS_API rtError_t MockRtKernelLaunchWithHandleV2(void *hdl, const uint64_t tilingKey, uint32_t blockDim,
    rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stm, const rtTaskCfgInfo_t *cfgInfo)
{
    std::cout << "Stub Func: " << __func__ << std::endl;
    return RT_ERROR_NONE;
}

RTS_API rtError_t MockRtKernelLaunchWithFlagV2(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo,
    rtSmDesc_t *smDesc, rtStream_t stm, uint32_t flags, const rtTaskCfgInfo_t *cfgInfo)
{
    std::cout << "Stub Func: " << __func__ << std::endl;
    return RT_ERROR_NONE;
}

ACL_FUNC_VISIBILITY aclError MockAclInit(const char *configPath)
{
    std::cout << "Stub Func: " << __func__ << std::endl;
    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError MockAclFinalize()
{
    std::cout << "Stub Func: " << __func__ << std::endl;
    return ACL_SUCCESS;
}

std::unordered_map<std::string, void *> g_funcMocks{
    {"rtKernelLaunch", reinterpret_cast<void *>(&MockRtKernelLaunch)},
    {"rtKernelLaunchWithHandleV2", reinterpret_cast<void *>(&MockRtKernelLaunchWithHandleV2)},
    {"rtKernelLaunchWithFlagV2", reinterpret_cast<void *>(&MockRtKernelLaunchWithFlagV2)},
    {"aclInit", reinterpret_cast<void *>(&MockAclInit)},
    {"aclFinalize", reinterpret_cast<void *>(&MockAclFinalize)},
};

bool g_isDlsymNullptr = false;

extern "C" {
void *dlopen(const char *filename, int flag)
{
    return reinterpret_cast<void *>(0x1234);
}

void *dlsym(void *handle, const char *symbol)
{
    if (g_isDlsymNullptr) {
        return nullptr;
    }
    return g_funcMocks.count(std::string(symbol)) ? g_funcMocks[std::string(symbol)] : nullptr;
}
}

TEST(AclHooks, do_aclInit_expect_success)
{
    g_isDlsymNullptr = false;
    const char *configPath = "test";
    aclInit(configPath);
    EXPECT_EQ(aclInit(configPath), ACL_SUCCESS);
}

TEST(AclHooks, do_aclFinalize_expect_success)
{
    g_isDlsymNullptr = false;
    EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
}

TEST(AclHooks, do_aclInit_expect_error)
{
    g_isDlsymNullptr = true;
    const char *configPath = "test";
    EXPECT_EQ(aclInit(configPath), ACL_ERROR_INTERNAL_ERROR);
}

TEST(AclHooks, do_aclFinalize_expect_error)
{
    g_isDlsymNullptr = true;
    EXPECT_EQ(aclFinalize(), ACL_ERROR_INTERNAL_ERROR);
}

TEST(RuntimeHooks, do_rtKernelLaunch_expect_success)
{
    g_isDlsymNullptr = false;
    const void *stubFunc = nullptr;
    uint32_t blockDim = 1;
    void *args = nullptr;
    uint32_t argsSize = 1;
    rtSmDesc_t *smDesc = nullptr;
    rtStream_t stm = nullptr;
    EXPECT_EQ(rtKernelLaunch(stubFunc, blockDim, args, argsSize, smDesc, stm), RT_ERROR_NONE);
}

TEST(RuntimeHooks, do_rtKernelLaunchWithHandleV2_expect_success)
{
    g_isDlsymNullptr = false;
    void *hdl = nullptr;
    const uint64_t tilingKey = 1;
    uint32_t blockDim = 1;
    rtArgsEx_t *argsInfo = nullptr;
    rtSmDesc_t *smDesc = nullptr;
    rtStream_t stm = nullptr;
    const rtTaskCfgInfo_t *cfgInfo = nullptr;
    EXPECT_EQ(rtKernelLaunchWithHandleV2(hdl, tilingKey, blockDim, argsInfo, smDesc, stm, cfgInfo), RT_ERROR_NONE);
}

TEST(RuntimeHooks, do_rtKernelLaunchWithFlagV2_expect_success)
{
    g_isDlsymNullptr = false;
    const void *stubFunc = nullptr;
    uint32_t blockDim = 1;
    rtArgsEx_t *argsInfo = nullptr;
    rtSmDesc_t *smDesc = nullptr;
    rtStream_t stm = nullptr;
    uint32_t flags = 1;
    const rtTaskCfgInfo_t *cfgInfo = nullptr;
    EXPECT_EQ(rtKernelLaunchWithFlagV2(stubFunc, blockDim, argsInfo, smDesc, stm, flags, cfgInfo), RT_ERROR_NONE);
}

TEST(RuntimeHooks, do_rtKernelLaunch_expect_error)
{
    g_isDlsymNullptr = true;
    const void *stubFunc = nullptr;
    uint32_t blockDim = 1;
    void *args = nullptr;
    uint32_t argsSize = 1;
    rtSmDesc_t *smDesc = nullptr;
    rtStream_t stm = nullptr;
    EXPECT_EQ(rtKernelLaunch(stubFunc, blockDim, args, argsSize, smDesc, stm), RT_ERROR_RESERVED);
}

TEST(RuntimeHooks, do_rtKernelLaunchWithHandleV2_expect_error)
{
    g_isDlsymNullptr = true;
    void *hdl = nullptr;
    const uint64_t tilingKey = 1;
    uint32_t blockDim = 1;
    rtArgsEx_t *argsInfo = nullptr;
    rtSmDesc_t *smDesc = nullptr;
    rtStream_t stm = nullptr;
    const rtTaskCfgInfo_t *cfgInfo = nullptr;
    EXPECT_EQ(rtKernelLaunchWithHandleV2(hdl, tilingKey, blockDim, argsInfo, smDesc, stm, cfgInfo), RT_ERROR_RESERVED);
}

TEST(RuntimeHooks, do_rtKernelLaunchWithFlagV2_expect_error)
{
    g_isDlsymNullptr = true;
    const void *stubFunc = nullptr;
    uint32_t blockDim = 1;
    rtArgsEx_t *argsInfo = nullptr;
    rtSmDesc_t *smDesc = nullptr;
    rtStream_t stm = nullptr;
    uint32_t flags = 1;
    const rtTaskCfgInfo_t *cfgInfo = nullptr;
    EXPECT_EQ(rtKernelLaunchWithFlagV2(stubFunc, blockDim, argsInfo, smDesc, stm, flags, cfgInfo), RT_ERROR_RESERVED);
}