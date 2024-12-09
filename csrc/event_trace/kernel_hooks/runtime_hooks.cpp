// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "runtime_hooks.h"

#include <cstdint>
#include <elf.h>
#include <vector>
#include <algorithm>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <unordered_map>
#include <map>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include "event_report.h"
#include "vallina_symbol.h"
#include "serializer.h"
#include "log.h"
#include "record_info.h"
#include "ustring.h"
#include "umask_guard.h"
#include "securec.h"
#include "global_handle.h"

using namespace Leaks;

namespace {

struct RuntimeLibLoader {
    static void *Load(void)
    {
        return dlopen("libruntime.so", RTLD_NOW | RTLD_GLOBAL);
    }
};
using RuntimeSymbol = VallinaSymbol<RuntimeLibLoader>;

std::vector<char *> ToRawCArgv(std::vector<std::string> const &argv)
{
    std::vector<char *> rawArgv;
    for (auto const &arg: argv) {
        rawArgv.emplace_back(const_cast<char *>(arg.data()));
    }
    rawArgv.emplace_back(nullptr);
    return rawArgv;
}
}  // namespace

bool PipeCall(std::vector<std::string> const &cmd, std::string &output)
{
    int pipeStdout[2];
    if (pipe(pipeStdout) != 0) {
        Utility::LogError("PipeCall: get pipe failed");
        return false;
    }

    pid_t pid = fork();
    if (pid < 0) {
        Utility::LogError("PipeCall: create subprocess failed");
        return false;
    } else if (pid == 0) {
        dup2(pipeStdout[1], STDOUT_FILENO);
        close(pipeStdout[0]);
        close(pipeStdout[1]);
        execvp(cmd[0].c_str(), ToRawCArgv(cmd).data());
        _exit(EXIT_FAILURE);
    } else {
        close(pipeStdout[1]);
        static constexpr std::size_t bufLen = 256UL;
        char buf[bufLen] = {'\0'};
        ssize_t nBytes = 0L;
        for (; (nBytes = read(pipeStdout[0], buf, bufLen)) > 0L;) {
            output.append(buf, static_cast<std::size_t>(nBytes));
        }
        close(pipeStdout[0]);
        int status;
        waitpid(pid, &status, 0);
        return WIFEXITED(status) && WEXITSTATUS(status) == 0;
    }
    return true;
}

std::string ParseLine(std::string const &line)
{
    std::vector<std::string> items;
    Utility::Split(line, std::back_inserter(items), " ");
    if (items.size() < 5UL) {
        return "";
    }
    constexpr std::size_t scopeIdx = 1UL;
    constexpr std::size_t symbolKindIdx = 2UL;
    if (items[scopeIdx] != "g" || items[symbolKindIdx] != "F") {
        return "";
    }
    constexpr std::size_t kernelNameIdx = 4UL;
    std::string kernelName = items[kernelNameIdx];
    if (Utility::EndWith(kernelName, "_mix_aic") ||
        Utility::EndWith(kernelName, "_mix_aiv")) {
        kernelName = kernelName.substr(0UL, kernelName.length() - 8UL);
    }

    items.clear();
    Utility::Split(kernelName, std::back_inserter(items), "_");
    if (items.size() < 2UL) {
        return "";
    }
    std::string kernelNamePrefix = items[0] + "_" + items[1];
    return kernelNamePrefix;
}

std::string ParseNameFromOutput(std::string output)
{
    std::string kernelName;
    std::vector<std::string> lines;
    Utility::Split(output, std::back_inserter(lines), "\n");

    // skip headers
    auto it = lines.cbegin();
    for (; it != lines.cend(); ++it) {
        if (it->find("SYMBOL TABLE:") != std::string::npos) {
            break;
        }
    }

    if (it == lines.cend()) {
        return kernelName;
    }
    ++it;

    for (; it != lines.cend(); ++it) {
        kernelName = ParseLine(*it);
        if (!kernelName.empty()) {
            return kernelName;
        }
    }
    return kernelName;
}

std::string GetNameFromBinary(void *hdl)
{
    std::string kernelName;
    auto it = GlobalHandle::GetInstance().GlobalHandle::GetInstance().handleBinKernelMap_.find(hdl);
    if (it == GlobalHandle::GetInstance().GlobalHandle::GetInstance().handleBinKernelMap_.end()) {
        Utility::LogError("kernel handle NOT registered in map");
        return kernelName;
    }
    std::vector<char> binary = it->second.bin;
    std::string kernelPath = "./kernel.o." + std::to_string(getpid());
    {
        Utility::UmaskGuard umaskGuard(REGULAR_MODE_MASK);
        if (!WriteBinary(kernelPath, binary.data(), binary.size())) {
            return kernelName;
        }
    }
    std::vector<std::string> cmd = {
        "llvm-objdump",
        "-t",
        kernelPath
    };

    std::string output;
    bool ret = PipeCall(cmd, output);
    kernelName = ParseNameFromOutput(output);
    remove(kernelPath.c_str());
    return kernelName;
}

std::string GetKernelNameByStubFunc(const void *stubFunc)
{
    std::string kernelName;
    auto it = GlobalHandle::GetInstance().stubHandleMap_.find(stubFunc);
    if (it == GlobalHandle::GetInstance().stubHandleMap_.end()) {
        Utility::LogError("stubFunc NOT registered in map");
        return kernelName;
    }
    kernelName = GetNameFromBinary(const_cast<void *>(it->second));
    return kernelName;
}

KernelLaunchRecord CreateKernelLaunchRecord(uint32_t blockDim, rtStream_t stm, KernelLaunchType type)
{
    auto record = KernelLaunchRecord {};
    int32_t *streamId = (int32_t*) malloc(sizeof(int32_t));
    rtGetStreamId(stm, streamId);
    record.type = type;
    record.blockDim = blockDim;
    record.streamId = *streamId;
    free(streamId);
    return record;
}

RTS_API rtError_t rtKernelLaunch(
    const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize, rtSmDesc_t *smDesc, rtStream_t stm)
{
    using RtKernelLaunch = decltype(&rtKernelLaunch);
    auto vallina = RuntimeSymbol::Instance().Get<RtKernelLaunch>(__func__);
    if (vallina == nullptr) {
        Utility::LogError("vallina func get FAILED");
        return RT_ERROR_RESERVED;
    }

    rtError_t ret = vallina(stubFunc, blockDim, args, argsSize, smDesc, stm);
    auto record = KernelLaunchRecord {};
    record = CreateKernelLaunchRecord(blockDim, stm, KernelLaunchType::NORMAL);
    if (EOK != strncpy_s(record.kernelName, sizeof(record.kernelName),
        GetKernelNameByStubFunc(stubFunc).c_str(), sizeof(record.kernelName) - 1)) {
        Utility::LogError("strncpy_s FAILED");
    }
    if (!EventReport::Instance(CommType::SOCKET).ReportKernelLaunch(record)) {
        Utility::LogError("%s report FAILED", __func__);
    }
    return ret;
}

RTS_API rtError_t rtKernelLaunchWithHandleV2(void *hdl, const uint64_t tilingKey, uint32_t blockDim,
    rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stm, const rtTaskCfgInfo_t *cfgInfo)
{
    using RtKernelLaunchWithHandleV2 = decltype(&rtKernelLaunchWithHandleV2);
    auto vallina = RuntimeSymbol::Instance().Get<RtKernelLaunchWithHandleV2>(__func__);
    if (vallina == nullptr) {
        Utility::LogError("vallina func get FAILED");
        return RT_ERROR_RESERVED;
    }

    rtError_t ret = vallina(hdl, tilingKey, blockDim, argsInfo, smDesc, stm, cfgInfo);
    auto record = KernelLaunchRecord {};
    record = CreateKernelLaunchRecord(blockDim, stm, KernelLaunchType::HANDLEV2);
    if (EOK != strncpy_s(record.kernelName, sizeof(record.kernelName),
        GetNameFromBinary(hdl).c_str(), sizeof(record.kernelName) - 1)) {
        Utility::LogError("strncpy_s FAILED");
    }
    if (!EventReport::Instance(CommType::SOCKET).ReportKernelLaunch(record)) {
        Utility::LogError("%s report FAILED", __func__);
    }
    return ret;
}

RTS_API rtError_t rtKernelLaunchWithFlagV2(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo,
    rtSmDesc_t *smDesc, rtStream_t stm, uint32_t flags, const rtTaskCfgInfo_t *cfgInfo)
{
    using RtKernelLaunchWithFlagV2 = decltype(&rtKernelLaunchWithFlagV2);
    auto vallina = RuntimeSymbol::Instance().Get<RtKernelLaunchWithFlagV2>(__func__);
    if (vallina == nullptr) {
        Utility::LogError("vallina func get FAILED");
        return RT_ERROR_RESERVED;
    }

    rtError_t ret = vallina(stubFunc, blockDim, argsInfo, smDesc, stm, flags, cfgInfo);
    auto record = KernelLaunchRecord {};
    record = CreateKernelLaunchRecord(blockDim, stm, KernelLaunchType::FLAGV2);
    if (EOK != strncpy_s(record.kernelName, sizeof(record.kernelName),
        GetKernelNameByStubFunc(stubFunc).c_str(), sizeof(record.kernelName) - 1)) {
        Utility::LogError("strncpy_s FAILED");
    }
    if (!EventReport::Instance(CommType::SOCKET).ReportKernelLaunch(record)) {
        Utility::LogError("%s report FAILED", __func__);
    }
    return ret;
}

RTS_API rtError_t rtGetStreamId(rtStream_t stm, int32_t *streamId)
{
    using rtGetStreamId = decltype(&rtGetStreamId);
    auto vallina = RuntimeSymbol::Instance().Get<rtGetStreamId>(__func__);
    if (vallina == nullptr) {
        Utility::LogError("vallina func get FAILED");
        return RT_ERROR_RESERVED;
    }
    rtError_t ret = vallina(stm, streamId);
    return ret;
}

RTS_API rtError_t rtFunctionRegister(
    void *binHandle, const void *stubFunc, const char *stubName, const void *kernelInfoExt, uint32_t funcMode)
{
    using RtFunctionRegister = decltype(&rtFunctionRegister);
    auto vallina = RuntimeSymbol::Instance().Get<RtFunctionRegister>(__func__);
    if (vallina == nullptr) {
        Utility::LogError("vallina func get FAILED");
        return RT_ERROR_RESERVED;
    }
    rtError_t result = vallina(binHandle, stubFunc, stubName, kernelInfoExt, funcMode);
    GlobalHandle::GetInstance().stubHandleMap_[stubFunc] = binHandle;
    return result;
}

RTS_API rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **hdl)
{
    using RtDevBinaryRegister = decltype(&rtDevBinaryRegister);
    auto vallina = RuntimeSymbol::Instance().Get<RtDevBinaryRegister>(__func__);
    if (vallina == nullptr) {
        Utility::LogError("vallina func get FAILED");
        return RT_ERROR_RESERVED;
    }
    
    rtError_t result = vallina(bin, hdl);
    if (result == RT_ERROR_NONE && bin != nullptr && bin->data != nullptr && hdl != nullptr) {
        // register handle bin map
        if (bin->length > MAX_BINARY_SIZE) {
            Utility::LogError("Illegal binary size: binary size[%u] exceeds max binary size[%u].",
                bin->length, MAX_BINARY_SIZE);
            return RT_ERROR_MEMORY_ALLOCATION ;
        }
        auto binData = static_cast<char const *>(bin->data);
        BinKernel binKernel {};
        binKernel.bin = std::vector<char>(binData, binData + bin->length);
        GlobalHandle::GetInstance().handleBinKernelMap_[*hdl] = std::move(binKernel);
    }
    return result;
}

RTS_API rtError_t rtRegisterAllKernel(const rtDevBinary_t *bin, void **hdl)
{
    using RtRegisterAllKernel = decltype(&rtRegisterAllKernel);
    auto vallina = RuntimeSymbol::Instance().Get<RtRegisterAllKernel>(__func__);
    if (vallina == nullptr) {
        Utility::LogError("vallina func get FAILED");
        return RT_ERROR_RESERVED;
    }
    rtError_t result = vallina(bin, hdl);
    if (result == RT_ERROR_NONE && bin != nullptr && bin->data != nullptr && hdl != nullptr) {
        // register handle bin map
        if (bin->length > MAX_BINARY_SIZE) {
            Utility::LogError("Illegal binary size: binary size[%u] exceeds max binary size[%u].",
                bin->length, MAX_BINARY_SIZE);
            return RT_ERROR_MEMORY_ALLOCATION ;
        }
        auto binData = static_cast<char const *>(bin->data);
        BinKernel binKernel {};
        binKernel.bin = std::vector<char>(binData, binData + bin->length);
        GlobalHandle::GetInstance().handleBinKernelMap_[*hdl] = std::move(binKernel);
    }
    return result;
}

RTS_API rtError_t rtDevBinaryUnRegister(void *hdl)
{
    using RtDevBinaryUnRegister = decltype(&rtDevBinaryUnRegister);
    auto vallina = RuntimeSymbol::Instance().Get<RtDevBinaryUnRegister>(__func__);
    if (vallina == nullptr) {
        Utility::LogError("vallina func get FAILED");
        return RT_ERROR_RESERVED;
    }

    rtError_t result = vallina(hdl);
    if (result == RT_ERROR_NONE) {
        // unregister handle bin map
        auto it = GlobalHandle::GetInstance().handleBinKernelMap_.find(hdl);
        if (it != GlobalHandle::GetInstance().handleBinKernelMap_.end()) {
            GlobalHandle::GetInstance().handleBinKernelMap_.erase(hdl);
        }
        // unregister stub handle map
        for (auto it = GlobalHandle::GetInstance().stubHandleMap_.begin();
             it != GlobalHandle::GetInstance().stubHandleMap_.end();) {
            if (it->second == hdl) {
                it = GlobalHandle::GetInstance().stubHandleMap_.erase(it);
            } else {
                ++it;
            }
        }
    }
    return result;
}