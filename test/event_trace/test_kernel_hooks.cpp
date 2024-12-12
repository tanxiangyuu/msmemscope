// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include <gtest/gtest.h>
#include <unordered_map>
#include "kernel_hooks/runtime_hooks.h"
#include "kernel_hooks/acl_hooks.h"
#include "vallina_symbol.h"
#include "handle_mapping.h"

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

RTS_API rtError_t MockRtGetStreamId(rtStream_t stm, int32_t *streamId)
{
    std::cout << "Stub Func: " << __func__ << std::endl;
    return RT_ERROR_NONE;
}

RTS_API rtError_t MockRtFunctionRegister(
    void *binHandle, const void *stubFunc, const char *stubName, const void *kernelInfoExt, uint32_t funcMode)
{
    std::cout << "Stub Func: " << __func__ << std::endl;
    return RT_ERROR_NONE;
}

RTS_API rtError_t MockRtDevBinaryRegister(const rtDevBinary_t *bin, void **hdl)
{
    std::cout << "Stub Func: " << __func__ << std::endl;
    return RT_ERROR_NONE;
}

RTS_API rtError_t MockRtRegisterAllKernel(const rtDevBinary_t *bin, void **hdl)
{
    std::cout << "Stub Func: " << __func__ << std::endl;
    return RT_ERROR_NONE;
}

RTS_API rtError_t MockRtDevBinaryUnRegister(void *hdl)
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
    {"rtGetStreamId", reinterpret_cast<void *>(&MockRtGetStreamId)},
    {"rtFunctionRegister", reinterpret_cast<void *>(&MockRtFunctionRegister)},
    {"rtDevBinaryRegister", reinterpret_cast<void *>(&MockRtDevBinaryRegister)},
    {"rtRegisterAllKernel", reinterpret_cast<void *>(&MockRtRegisterAllKernel)},
    {"rtDevBinaryUnRegister", reinterpret_cast<void *>(&MockRtDevBinaryUnRegister)},
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

TEST(RuntimeHooks, do_rtGetStreamId_expect_success)
{
    g_isDlsymNullptr = false;
    rtStream_t stm = nullptr;
    int32_t *streamId = nullptr;
    EXPECT_EQ(rtGetStreamId(stm, streamId), RT_ERROR_NONE);
}

TEST(RuntimeHooks, do_rtGetStreamId_expect_error)
{
    g_isDlsymNullptr = true;
    rtStream_t stm = nullptr;
    int32_t *streamId = nullptr;
    EXPECT_EQ(rtGetStreamId(stm, streamId), RT_ERROR_RESERVED);
}

TEST(RuntimeHooks, do_rtFunctionRegister_expect_success)
{
    g_isDlsymNullptr = false;
    void *binHandle = nullptr;
    const void *stubFunc = nullptr;
    const char *stubName = nullptr;
    const void *kernelInfoExt = nullptr;
    uint32_t funcMode = 1;
    EXPECT_EQ(rtFunctionRegister(binHandle, stubFunc, stubName, kernelInfoExt, funcMode), RT_ERROR_NONE);
}

TEST(RuntimeHooks, do_rtFunctionRegister_expect_error)
{
    g_isDlsymNullptr = true;
    void *binHandle = nullptr;
    const void *stubFunc = nullptr;
    const char *stubName = nullptr;
    const void *kernelInfoExt = nullptr;
    uint32_t funcMode = 1;
    EXPECT_EQ(rtFunctionRegister(binHandle, stubFunc, stubName, kernelInfoExt, funcMode), RT_ERROR_RESERVED);
}

TEST(RuntimeHooks, do_rtDevBinaryRegister_expect_success)
{
    g_isDlsymNullptr = false;
    std::string mockData = ("SYMBOL TABLE:\n"
    "000 g F .text test_000_mix_aic"
    "000 g O .data g_opSystemRunCfg\n");
    rtDevBinary_t bin;
    bin.data = mockData.data();
    bin.length = mockData.length();
    std::vector<uint8_t> handleData{1, 2, 3};
    void *hdl = handleData.data();
    EXPECT_EQ(rtDevBinaryRegister(&bin, &hdl), RT_ERROR_NONE);
}

TEST(RuntimeHooks, do_rtDevBinaryRegister_expect_error)
{
    g_isDlsymNullptr = true;
    const rtDevBinary_t *bin = nullptr;
    void **hdl = nullptr;
    EXPECT_EQ(rtDevBinaryRegister(bin, hdl), RT_ERROR_RESERVED);
}

TEST(RuntimeHooks, do_rtDevBinaryRegister_exceeds_binary_size_expect_error)
{
    g_isDlsymNullptr = false;
    std::string mockData = ("SYMBOL TABLE:\n"
    "000 g F .text test_000_mix_aic"
    "000 g O .data g_opSystemRunCfg\n");
    rtDevBinary_t bin;
    bin.data = mockData.data();
    bin.length = 32ULL * 1024 * 1024 * 1024 + 1;
    std::vector<uint8_t> handleData{1, 2, 3};
    void *hdl = handleData.data();
    EXPECT_EQ(rtDevBinaryRegister(&bin, &hdl), RT_ERROR_MEMORY_ALLOCATION);
}

TEST(RuntimeHooks, do_rtRegisterAllKernel_expect_success)
{
    g_isDlsymNullptr = false;
    std::string mockData = ("SYMBOL TABLE:\n"
    "000 g F .text test_000_mix_aic"
    "000 g O .data g_opSystemRunCfg\n");
    rtDevBinary_t bin;
    bin.data = mockData.data();
    bin.length = mockData.length();
    std::vector<uint8_t> handleData{1, 2, 3};
    void *hdl = handleData.data();
    EXPECT_EQ(rtRegisterAllKernel(&bin, &hdl), RT_ERROR_NONE);
}

TEST(RuntimeHooks, do_rtRegisterAllKernel_expect_error)
{
    g_isDlsymNullptr = true;
    const rtDevBinary_t *bin = nullptr;
    void **hdl = nullptr;
    EXPECT_EQ(rtRegisterAllKernel(bin, hdl), RT_ERROR_RESERVED);
}

TEST(RuntimeHooks, do_rtRegisterAllKernel_exceeds_binary_size_expect_error)
{
    g_isDlsymNullptr = false;
    std::string mockData = ("SYMBOL TABLE:\n"
    "000 g F .text test_000_mix_aic"
    "000 g O .data g_opSystemRunCfg\n");
    rtDevBinary_t bin;
    bin.data = mockData.data();
    bin.length = 32ULL * 1024 * 1024 * 1024 + 1;
    std::vector<uint8_t> handleData{1, 2, 3};
    void *hdl = handleData.data();
    EXPECT_EQ(rtRegisterAllKernel(&bin, &hdl), RT_ERROR_MEMORY_ALLOCATION);
}

TEST(RuntimeHooks, do_rtDevBinaryUnRegister_expect_success)
{
    g_isDlsymNullptr = false;
    std::vector<uint8_t> handleData{1, 2, 3};
    void *hdl = handleData.data();
    void *stubFunc = handleData.data();
    Leaks::BinKernel binkernel {};
    binkernel.bin = {0x01, 0x02, 0x03, 0x04};
    Leaks::HandleMapping::GetInstance().handleBinKernelMap_.insert({hdl, binkernel});
    Leaks::HandleMapping::GetInstance().stubHandleMap_.insert({stubFunc, hdl});
    EXPECT_EQ(rtDevBinaryUnRegister(hdl), RT_ERROR_NONE);
}

TEST(RuntimeHooks, do_rtDevBinaryUnRegister_expect_error)
{
    g_isDlsymNullptr = true;
    void *hdl = nullptr;
    EXPECT_EQ(rtDevBinaryUnRegister(hdl), RT_ERROR_RESERVED);
}

TEST(KernelNameFunc, do_pipe_call_with_ls_return_correct_output)
{
    std::vector<std::string> argv = {"/bin/ls", "/tmp"};
    std::string output;
    ASSERT_TRUE(PipeCall(argv, output));
    ASSERT_FALSE(output.empty());
}

TEST(KernelNameFunc, pipe_call_with_test_command_return_false)
{
    std::vector<std::string> argv = {"test-command"};
    std::string output;
    ASSERT_FALSE(PipeCall(argv, output));
}

TEST(KernelNameFunc, write_binary_with_vaild_data_return_success_and_read_correct_data)
{
    std::string fileName = "./test.bin";
    char wbuf[] = "123456789";
    ASSERT_TRUE(WriteBinary(fileName, wbuf, sizeof(wbuf)));

    std::ifstream ifs;
    ifs.open(fileName, std::ios::binary);
    ASSERT_TRUE(ifs.is_open());
    char rbuf[sizeof(wbuf)];
    ifs.read(rbuf, sizeof(rbuf));
    ifs.close();
    ASSERT_EQ(std::string(wbuf), std::string(rbuf));

    std::remove(fileName.c_str());
}

TEST(KernelNameFunc, parseLine_with_kernelName_line_return_true_kernelName)
{
    std::string line = "0000 g F .text ReduceSum_7a09f_high_performance_123_mix_aiv";
    std::string kernelName;
    kernelName = ParseLine(line);
    ASSERT_EQ(kernelName, "ReduceSum_7a09f");
}

TEST(KernelNameFunc, do_parseLine_with_not_kernelName_line_return_empty_name)
{
    std::string line = "000 g O .data g_opSystemRunCfg";
    std::string kernelName;
    kernelName = ParseLine(line);
    ASSERT_EQ(kernelName, "");
}

TEST(KernelNameFunc, parseNameFromOutput_with_symbol_table_output_return_true_kernelName)
{
    std::string output = ("SYMBOL TABLE:\n"
    "000 g F .text test_000_mix_aic"
    "000 g O .data g_opSystemRunCfg\n");
    std::string kernelName;
    kernelName = ParseNameFromOutput(output);
    ASSERT_EQ(kernelName, "test_000");
}

TEST(KernelNameFunc, parseNameFromOutput_with_invaild_symbol_table_output_return_empty_name)
{
    std::string output = ("TEST TABLE:\n"
    "000 g F .text test_000_mix_aic"
    "000 g O .data g_opSystemRunCfg\n");
    std::string kernelName;
    kernelName = ParseNameFromOutput(output);
    ASSERT_EQ(kernelName, "");
}

TEST(KernelNameFunc, getNameFromBinary_with_hdl_return_empty_kernelName)
{
    std::string output = ("SYMBOL TABLE:\n"
    "000 g F .text test_000_mix_aic"
    "000 g O .data g_opSystemRunCfg\n");
    std::vector<uint8_t> handleData{1, 2, 3};
    void *hdl = handleData.data();
    Leaks::BinKernel binData {};
    binData.bin = {0x01, 0x02, 0x03, 0x04};
    std::string kernelName;
    Leaks::HandleMapping::GetInstance().handleBinKernelMap_.insert({hdl, binData});
    kernelName = GetNameFromBinary(hdl);
    Leaks::HandleMapping::GetInstance().handleBinKernelMap_.erase(hdl);
    ASSERT_EQ(kernelName, "");
}

TEST(KernelNameFunc, getKernelNameByStubFunc_with_stubfunc_return_empty_kernelName)
{
    std::string kernelName;
    std::vector<uint8_t> handleData{1, 2, 3};
    void *hdl = handleData.data();
    void *stubFunc = handleData.data();
    Leaks::HandleMapping::GetInstance().stubHandleMap_.insert({stubFunc, hdl});
    kernelName = GetKernelNameByStubFunc(stubFunc);
    Leaks::HandleMapping::GetInstance().stubHandleMap_.erase(stubFunc);
    ASSERT_EQ(kernelName, "");
}