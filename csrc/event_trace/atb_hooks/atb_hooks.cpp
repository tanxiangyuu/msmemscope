// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "atb_hooks.h"

#include <cstdio>
#include <dlfcn.h>
#include <sstream>
#include <mutex>
#include <vector>

#include "event_report.h"
#include "bit_field.h"
#include "securec.h"
#include "op_watch/op_excute_watch.h"
#include "trace_manager/event_trace_manager.h"

using namespace Leaks;

namespace atb {
    static std::string LeaksGetTensorInfo(const atb::Tensor& tensor)
    {
        std::ostringstream oss;
        oss << "dtype:" << LeaksEnumToString(tensor.desc.dtype)
            << ",format:" << LeaksEnumToString(tensor.desc.format)
            << ",shape:[";
        for (size_t i = 0; i < tensor.desc.shape.dimNum; i++) {
            oss << tensor.desc.shape.dims[i] << ",";
        }
        oss << "]";
        return oss.str();
    }

    static std::string LeaksGetTensorInfo(const Mki::Tensor& tensor)
    {
        std::ostringstream oss;
        oss << "dtype:" << LeaksEnumToString(tensor.desc.dtype)
            << ",format:" << LeaksEnumToString(tensor.desc.format)
            << ",shape:[";
        for (auto& dim : tensor.desc.dims) {
            oss << dim << ",";
        }
        oss << "]";
        return oss.str();
    }

    static std::string LeaksGetOpParams(const atb::RunnerVariantPack& runnerVariantPack, const std::string& path)
    {
        std::ostringstream oss;
        oss << "path:" << path << ",workspace_ptr:"
            << static_cast<void*>(runnerVariantPack.workspaceBuffer) << ",workspace_size:"
            << Utility::GetAddResult(runnerVariantPack.workspaceBufferSize, runnerVariantPack.intermediateBufferSize);
        return oss.str();
    }

    static void LeaksReportTensors(const atb::RunnerVariantPack& runnerVariantPack, const std::string& name)
    {
        std::vector<RecordBuffer> buffers;
        for (auto& tensor : runnerVariantPack.inTensors) {
            RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<MemAccessRecord>(
                TLVBlockType::OP_NAME, name, TLVBlockType::MEM_ATTR, LeaksGetTensorInfo(tensor));
            MemAccessRecord* record = buffer.Cast<MemAccessRecord>();
            record->addr = static_cast<uint64_t>((std::uintptr_t)tensor.deviceData);
            record->memSize = tensor.dataSize;
            record->eventType = AccessType::UNKNOWN;
            record->memType = AccessMemType::ATB;
            buffers.push_back(buffer);
        }
        for (auto& tensor : runnerVariantPack.outTensors) {
            RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<MemAccessRecord>(
                TLVBlockType::OP_NAME, name, TLVBlockType::MEM_ATTR, LeaksGetTensorInfo(tensor));
            MemAccessRecord* record = buffer.Cast<MemAccessRecord>();
            record->addr = static_cast<uint64_t>((std::uintptr_t)tensor.deviceData);
            record->memSize = tensor.dataSize;
            record->eventType = AccessType::WRITE;
            record->memType = AccessMemType::ATB;
            buffers.push_back(buffer);
        }

        if (!EventReport::Instance(LeaksCommType::SHARED_MEMORY).ReportAtbAccessMemory(buffers)) {
            CLIENT_ERROR_LOG("Report atb op end event failed.\n");
        }
        return;
    }

    static void LeaksReportTensors(Mki::LeaksOriginalGetInTensors &getInTensors,
        Mki::LeaksOriginalGetOutTensors &getOutTensors,
        const Mki::LaunchParam &launchParam, const std::string& name)
    {
        std::vector<RecordBuffer> buffers;
        for (auto& tensor : getInTensors(const_cast<Mki::LaunchParam*>(&launchParam))) {
            RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<MemAccessRecord>(
                TLVBlockType::OP_NAME, name, TLVBlockType::MEM_ATTR, LeaksGetTensorInfo(tensor));
            MemAccessRecord* record = buffer.Cast<MemAccessRecord>();
            record->addr = static_cast<uint64_t>((std::uintptr_t)tensor.data);
            record->memSize = tensor.dataSize;
            record->eventType = AccessType::UNKNOWN;
            record->memType = AccessMemType::ATB;
            buffers.push_back(buffer);
        }
        for (auto& tensor : getOutTensors(const_cast<Mki::LaunchParam*>(&launchParam))) {
            RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<MemAccessRecord>(
                TLVBlockType::OP_NAME, name, TLVBlockType::MEM_ATTR, LeaksGetTensorInfo(tensor));
            MemAccessRecord* record = buffer.Cast<MemAccessRecord>();
            record->addr = static_cast<uint64_t>((std::uintptr_t)tensor.data);
            record->memSize = tensor.dataSize;
            record->eventType = AccessType::WRITE;
            record->memType = AccessMemType::ATB;
            buffers.push_back(buffer);
        }

        if (!EventReport::Instance(LeaksCommType::SHARED_MEMORY).ReportAtbAccessMemory(buffers)) {
            CLIENT_ERROR_LOG("Report atb op end event failed.\n");
        }
        return;
    }

    static void LeaksReportOp(const std::string& name, const std::string& params, bool isStart)
    {
        RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<AtbOpExecuteRecord>(
            TLVBlockType::ATB_NAME, name, TLVBlockType::ATB_PARAMS, params);
        AtbOpExecuteRecord* record = buffer.Cast<AtbOpExecuteRecord>();
        record->subtype = isStart ? RecordSubType::ATB_START : RecordSubType::ATB_END;

        if (!EventReport::Instance(LeaksCommType::SHARED_MEMORY).ReportAtbOpExecute(buffer)) {
            CLIENT_ERROR_LOG("Report atb op start event failed.\n");
        }
    }

    static bool LeaksGetOpNameAndDir(atb::Runner* thisPtr, std::string& name, std::string& dir)
    {
#if defined(_GLIBCXX_USE_CXX11_ABI) && (_GLIBCXX_USE_CXX11_ABI == 0)
        static auto funcGetOperationName = VallinaSymbol<ATBLibLoader>::Instance().Get<LeaksOriginalGetOperationName>(
            "_ZNK3atb6Runner16GetOperationNameEv");
        static auto funcGetSaveTensorDir = VallinaSymbol<ATBLibLoader>::Instance().Get<LeaksOriginalGetSaveTensorDir>(
            "_ZNK3atb6Runner16GetSaveTensorDirEv");
#else
        static auto funcGetOperationName = VallinaSymbol<ATBLibLoader>::Instance().Get<LeaksOriginalGetOperationName>(
            "_ZNK3atb6Runner16GetOperationNameB5cxx11Ev");
        static auto funcGetSaveTensorDir = VallinaSymbol<ATBLibLoader>::Instance().Get<LeaksOriginalGetSaveTensorDir>(
            "_ZNK3atb6Runner16GetSaveTensorDirB5cxx11Ev");
#endif
        if (funcGetOperationName == nullptr || funcGetSaveTensorDir == nullptr) {
            CLIENT_ERROR_LOG("Cannot find origin function of atb.\n");
            return false;
        }
        name = funcGetOperationName(thisPtr);
        dir = funcGetSaveTensorDir(thisPtr);
        return true;
    }

    static bool LeaksGetAclrtStream(atb::Runner* thisPtr, const atb::RunnerVariantPack& runnerVariantPack,
        aclrtStream &stream)
    {
        static auto funcGetExecuteStream = VallinaSymbol<ATBLibLoader>::Instance().Get<LeaksOriginalGetExecuteStream>(
            "_ZNK3atb6Runner16GetExecuteStreamEPNS_7ContextE");
        if (funcGetExecuteStream == nullptr) {
            CLIENT_ERROR_LOG("Cannot find origin function of atb.\n");
            return false;
        }
        stream = funcGetExecuteStream(thisPtr, runnerVariantPack.context);
        return true;
    }

    static void LeaksParseKernelPath(bool& isBeforeLaunch, std::string& name, const std::string& dirPath)
    {
        auto beforePos = dirPath.find("/before");
        auto afterPos = dirPath.find("/after");
        isBeforeLaunch = true;
        std::string path;
        if (beforePos != std::string::npos) {
            path = dirPath.substr(0, beforePos);
        } else if (afterPos != std::string::npos) {
            isBeforeLaunch = false;
            path = dirPath.substr(0, afterPos);
        } else {
            name = "INVALID";
            CLIENT_ERROR_LOG("Cannot get kernel path.\n");
            return;
        }

        name = path;
        size_t lastSlashPos = name.find_last_of('/');
        if (lastSlashPos != std::string::npos) {
            name = name.substr(lastSlashPos + 1);
        }
    }

    static bool LeaksReportAtbKernel(const std::string& name, const std::string& dirPath, const bool& isBeforeLaunch)
    {
        if (name == "INVALID") {
            return false;
        }

        std::ostringstream oss;
        oss << "path:" << dirPath;
        std::string params = oss.str();

        RecordBuffer buffer = RecordBuffer::CreateRecordBuffer<AtbKernelRecord>(
            TLVBlockType::ATB_NAME, name, TLVBlockType::ATB_PARAMS, params);

        AtbKernelRecord* record = buffer.Cast<AtbKernelRecord>();
        record->subtype = isBeforeLaunch ? RecordSubType::KERNEL_START : RecordSubType::KERNEL_END;

        if (!EventReport::Instance(LeaksCommType::SHARED_MEMORY).ReportAtbKernel(buffer)) {
            CLIENT_ERROR_LOG("Report atb run kernel event failed.\n");
        }
        return true;
    }
}

extern "C" atb::Status _ZN3atb6Runner7ExecuteERNS_17RunnerVariantPackE(atb::Runner* thisPtr,
    atb::RunnerVariantPack& runnerVariantPack)
{
    static auto funcExecute = VallinaSymbol<ATBLibLoader>::Instance().Get<atb::LeaksOriginalRunnerExecuteFunc>(
        "_ZN3atb6Runner7ExecuteERNS_17RunnerVariantPackE");
    if (funcExecute == nullptr) {
        CLIENT_ERROR_LOG("Cannot find origin function of atb.\n");
        return 0;
    }
    if (!EventTraceManager::Instance().IsNeedTrace(RecordType::ATB_OP_EXECUTE_RECORD)) {
        return funcExecute(thisPtr, runnerVariantPack);
    }
    std::string params;
    std::string name;
    std::string dir;
    aclrtStream stream;
    if (!atb::LeaksGetOpNameAndDir(thisPtr, name, dir)
        || !atb::LeaksGetAclrtStream(thisPtr, runnerVariantPack, stream)) {
        return 0;
    }
    if (EventTraceManager::Instance().IsNeedTrace(RecordType::OP_LAUNCH_RECORD)) {
        params = atb::LeaksGetOpParams(runnerVariantPack, dir);
        atb::LeaksReportOp(name, params, true);
    }
    if (EventTraceManager::Instance().IsNeedTrace(RecordType::MEM_ACCESS_RECORD)) {
        atb::LeaksReportTensors(runnerVariantPack, name);
    }
    char cDirPath[WATCH_OP_DIR_MAX_LENGTH];
    static Config config = GetConfig();
    if (config.watchConfig.isWatched) {
        if (strncpy_s(cDirPath, WATCH_OP_DIR_MAX_LENGTH, dir.c_str(), WATCH_OP_DIR_MAX_LENGTH - 1) != EOK) {
            CLIENT_ERROR_LOG("strncpy_s FAILED");
            cDirPath[0] = '\0';
        }
    }
    if (config.watchConfig.isWatched) {
        OpExcuteBegin(stream, cDirPath, AccessMemType::ATB);
    }
    atb::Status st = funcExecute(thisPtr, runnerVariantPack);
    if (config.watchConfig.isWatched) {
        MonitoredTensor tensors[runnerVariantPack.outTensors.size()];
        size_t loop = 0;
        for (auto &item : runnerVariantPack.outTensors) {
            tensors[loop].dataSize = item.dataSize;
            tensors[loop].data = item.deviceData;
            loop++;
        }
        OpExcuteEnd(stream, cDirPath, tensors, runnerVariantPack.outTensors.size(), AccessMemType::ATB);
    }
    if (EventTraceManager::Instance().IsNeedTrace(RecordType::OP_LAUNCH_RECORD)) {
        atb::LeaksReportOp(name, params, false);
    }
    if (EventTraceManager::Instance().IsNeedTrace(RecordType::MEM_ACCESS_RECORD)) {
        atb::LeaksReportTensors(runnerVariantPack, name);
    }
    return st;
}

// 不调用原函数，原函数功能不与msleaks兼容
#if defined(_GLIBCXX_USE_CXX11_ABI) && (_GLIBCXX_USE_CXX11_ABI == 0)
extern "C" void _ZN3atb9StoreUtil15SaveLaunchParamEPvRKN3Mki11LaunchParamERKSs
#else
extern "C" void _ZN3atb9StoreUtil15SaveLaunchParamEPvRKN3Mki11LaunchParamERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
#endif
(aclrtStream stream, const Mki::LaunchParam& launchParam, const std::string& dirPath)
{
    if (!EventTraceManager::Instance().IsNeedTrace(RecordType::ATB_KERNEL_RECORD)) {
        return;
    }

    static auto getInTensors = VallinaSymbol<ATBLibLoader>::Instance().Get<Mki::LeaksOriginalGetInTensors>(
        "_ZN3Mki11LaunchParam12GetInTensorsEv");
    static auto getOutTensors = VallinaSymbol<ATBLibLoader>::Instance().Get<Mki::LeaksOriginalGetOutTensors>(
        "_ZN3Mki11LaunchParam13GetOutTensorsEv");
    if (getInTensors == nullptr || getOutTensors == nullptr) {
        CLIENT_ERROR_LOG("Cannot find origin function of atb.\n");
        return;
    }
    
    if (GetConfig().watchConfig.isWatched) {
        char cDirPath[WATCH_OP_DIR_MAX_LENGTH];
        if (strncpy_s(cDirPath, WATCH_OP_DIR_MAX_LENGTH, dirPath.c_str(), WATCH_OP_DIR_MAX_LENGTH - 1) != EOK) {
            CLIENT_ERROR_LOG("strncpy_s FAILED");
            cDirPath[0] = '\0';
        }
        KernelExcute(stream, cDirPath, getOutTensors(const_cast<Mki::LaunchParam*>(&launchParam)), AccessMemType::ATB);
    }
    std::string name;
    bool isBeforeLaunch;
    atb::LeaksParseKernelPath(isBeforeLaunch, name, dirPath);

    if (EventTraceManager::Instance().IsNeedTrace(RecordType::KERNEL_LAUNCH_RECORD)) {
        if (!atb::LeaksReportAtbKernel(name, dirPath, isBeforeLaunch)) {
            return;
        }
    }

    if (EventTraceManager::Instance().IsNeedTrace(RecordType::MEM_ACCESS_RECORD)) {
        atb::LeaksReportTensors(getInTensors, getOutTensors, launchParam, name);
    }
}

// 劫持判断函数，保证SaveLaunchParam函数可被调用
#if defined(_GLIBCXX_USE_CXX11_ABI) && (_GLIBCXX_USE_CXX11_ABI == 0)
extern "C" bool _ZN3atb5Probe16IsTensorNeedSaveERKSt6vectorIlSaIlEERKSs
#else
extern "C" bool _ZN3atb5Probe16IsTensorNeedSaveERKSt6vectorIlSaIlEERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
#endif
(const std::vector<int64_t>& ids, const std::string& opType)

{
    return EventTraceManager::Instance().IsNeedTrace(RecordType::ATB_KERNEL_RECORD);
}

extern "C" bool _ZN3atb5Probe17IsSaveTensorAfterEv()
{
    return true;
}

extern "C" bool _ZN3atb5Probe18IsSaveTensorBeforeEv()
{
    return true;
}

extern "C" bool _ZN3atb5Probe21IsExecuteCountInRangeEm(const uint64_t executeCount)
{
    return true;
}

// 劫持判断函数，保证path信息被配置
extern "C" bool _ZN3atb5Probe16IsSaveTensorDescEv()
{
    return true;
}

// 避免调用SaveVariantPack
extern "C" bool _ZNK3atb6Runner12IsSaveTensorEv()
{
    return false;
}