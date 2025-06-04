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

using namespace Leaks;

namespace atb {
    std::string LeaksGetTensorInfo(const atb::Tensor& tensor)
    {
        std::ostringstream oss;
        oss << "dtype:" << LeaksEnumToString(tensor.desc.dtype)
            << ",format:" << LeaksEnumToString(tensor.desc.format)
            << ",shape:";
        for (size_t i = 0; i < tensor.desc.shape.dimNum; i++) {
            oss << tensor.desc.shape.dims[i] << " ";
        }
        oss << ",type:atb tensor";
        return oss.str();
    }

    std::string LeaksGetTensorInfo(const Mki::Tensor& tensor)
    {
        std::ostringstream oss;
        oss << "dtype:" << LeaksEnumToString(tensor.desc.dtype)
            << ",format:" << LeaksEnumToString(tensor.desc.format)
            << ",shape:";
        for (auto& dim : tensor.desc.dims) {
            oss << dim << " ";
        }
        oss << ",type:kernel tensor";
        return oss.str();
    }

    void LeaksReportTensors(atb::RunnerVariantPack& runnerVariantPack, const std::string& name)
    {
        std::vector<MemAccessRecord> records;
        for (auto& tensor : runnerVariantPack.inTensors) {
            MemAccessRecord record;
            record.addr = static_cast<uint64_t>((std::uintptr_t)tensor.deviceData);
            record.memSize = tensor.dataSize;
            record.eventType = AccessType::UNKNOWN;
            record.memType = OpType::ATB;
            if (strncpy_s(record.name, sizeof(record.name), name.c_str(), sizeof(record.name) - 1) != EOK) {
                CLIENT_ERROR_LOG("strncpy_s FAILED");
                record.name[0] = '\0';
            }
            if (strncpy_s(record.attr, sizeof(record.attr),
                LeaksGetTensorInfo(tensor).c_str(), sizeof(record.attr) - 1) != EOK) {
                CLIENT_ERROR_LOG("strncpy_s FAILED");
                record.attr[0] = '\0';
            }
            records.push_back(record);
        }
        for (auto& tensor : runnerVariantPack.outTensors) {
            MemAccessRecord record;
            record.addr = static_cast<uint64_t>((std::uintptr_t)tensor.deviceData);
            record.memSize = tensor.dataSize;
            record.eventType = AccessType::WRITE;
            record.memType = OpType::ATB;
            if (strncpy_s(record.name, sizeof(record.name), name.c_str(), sizeof(record.name) - 1) != EOK) {
                CLIENT_ERROR_LOG("strncpy_s FAILED");
                record.name[0] = '\0';
            }
            if (strncpy_s(record.attr, sizeof(record.attr),
                LeaksGetTensorInfo(tensor).c_str(), sizeof(record.attr) - 1) != EOK) {
                CLIENT_ERROR_LOG("strncpy_s FAILED");
                record.attr[0] = '\0';
            }
            records.push_back(record);
        }

        if (!EventReport::Instance(CommType::SOCKET).ReportAtbAccessMemory(records)) {
            CLIENT_ERROR_LOG("Report atb op end event failed.\n");
        }
        return;
    }

    void LeaksReportTensors(Mki::LeaksOriginalGetInTensors &getInTensors, Mki::LeaksOriginalGetInTensors &getOutTensors,
        const Mki::LaunchParam &launchParam, const std::string& name)
    {
        std::vector<MemAccessRecord> records;
        for (auto& tensor : getInTensors(const_cast<Mki::LaunchParam*>(&launchParam))) {
            MemAccessRecord record;
            record.addr = static_cast<uint64_t>((std::uintptr_t)tensor.data);
            record.memSize = tensor.dataSize;
            record.eventType = AccessType::UNKNOWN;
            record.memType = OpType::ATB;
            if (strncpy_s(record.name, sizeof(record.name), name.c_str(), sizeof(record.name) - 1) != EOK) {
                CLIENT_ERROR_LOG("strncpy_s FAILED");
                record.name[0] = '\0';
            }
            if (strncpy_s(record.attr, sizeof(record.attr),
                LeaksGetTensorInfo(tensor).c_str(), sizeof(record.attr) - 1) != EOK) {
                CLIENT_ERROR_LOG("strncpy_s FAILED");
                record.attr[0] = '\0';
            }
            records.push_back(record);
        }
        for (auto& tensor : getOutTensors(const_cast<Mki::LaunchParam*>(&launchParam))) {
            MemAccessRecord record;
            record.addr = static_cast<uint64_t>((std::uintptr_t)tensor.data);
            record.memSize = tensor.dataSize;
            record.eventType = AccessType::WRITE;
            record.memType = OpType::ATB;
            if (strncpy_s(record.name, sizeof(record.name), name.c_str(), sizeof(record.name) - 1) != EOK) {
                CLIENT_ERROR_LOG("strncpy_s FAILED");
                record.name[0] = '\0';
            }
            if (strncpy_s(record.attr, sizeof(record.attr),
                LeaksGetTensorInfo(tensor).c_str(), sizeof(record.attr) - 1) != EOK) {
                CLIENT_ERROR_LOG("strncpy_s FAILED");
                record.attr[0] = '\0';
            }
            records.push_back(record);
        }

        if (!EventReport::Instance(CommType::SOCKET).ReportAtbAccessMemory(records)) {
            CLIENT_ERROR_LOG("Report atb op end event failed.\n");
        }
        return;
    }

    void LeaksReportOp(const std::string& name, const std::string& params, bool isStart)
    {
        AtbOpExecuteRecord record;
        record.eventType = isStart ? OpEventType::ATB_START : OpEventType::ATB_END;
        if (strncpy_s(record.name, sizeof(record.name), name.c_str(), sizeof(record.name) - 1) != EOK) {
            CLIENT_ERROR_LOG("strncpy_s FAILED");
            record.name[0] = '\0';
        }
        if (strncpy_s(record.params, sizeof(record.params), params.c_str(), sizeof(record.params) - 1) != EOK) {
            CLIENT_ERROR_LOG("strncpy_s FAILED");
            record.params[0] = '\0';
        }
        if (!EventReport::Instance(CommType::SOCKET).ReportAtbOpExecute(record)) {
            CLIENT_ERROR_LOG("Report atb op start event failed.\n");
        }
    }

    void GetOpNameAndDir(atb::Runner* thisPtr, std::string& name, std::string& dir)
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
            return ;
        }
        name = funcGetOperationName(thisPtr);
        dir = funcGetSaveTensorDir(thisPtr);
    }

    atb::Status LeaksRunnerExecute(atb::Runner* thisPtr, atb::RunnerVariantPack& runnerVariantPack)
    {
        std::string name;
        std::string dir;
        GetOpNameAndDir(thisPtr, name, dir);
        static auto funcExecute = VallinaSymbol<ATBLibLoader>::Instance().Get<LeaksOriginalRunnerExecuteFunc>(
            "_ZN3atb6Runner7ExecuteERNS_17RunnerVariantPackE");
        if (funcExecute == nullptr) {
            CLIENT_ERROR_LOG("Cannot find origin function of atb.\n");
            return 0;
        }
        Config config = EventReport::Instance(CommType::SOCKET).GetConfig();
        BitField<decltype(config.levelType)> levelType(config.levelType);
        if (!levelType.checkBit(static_cast<size_t>(LevelType::LEVEL_OP))) {
            return funcExecute(thisPtr, runnerVariantPack);
        }
        BitField<decltype(config.eventType)> eventType(config.eventType);
        std::string params;
        if (eventType.checkBit(static_cast<size_t>(EventType::LAUNCH_EVENT))) {
            std::ostringstream oss;
            oss << "path:" << dir << ",workspace ptr:"
                << static_cast<void*>(runnerVariantPack.workspaceBuffer) << ",workspace size:"
                << runnerVariantPack.workspaceBufferSize + runnerVariantPack.intermediateBufferSize;
            params = oss.str();
            atb::LeaksReportOp(name, params, true);
        }
        if (eventType.checkBit(static_cast<size_t>(EventType::ACCESS_EVENT))) {
            atb::LeaksReportTensors(runnerVariantPack, name);
        }
        if (config.watchConfig.isWatched) {
            Leaks::OpExcuteWatch::GetInstance().OpExcuteBegin(dir, OpType::ATB);
        }
        atb::Status st = funcExecute(thisPtr, runnerVariantPack);
        if (config.watchConfig.isWatched) {
            std::vector<MonitoredTensor> outputTensors;
            for (auto &item : runnerVariantPack.outTensors) {
                MonitoredTensor tensor {};
                tensor.data = item.deviceData;
                tensor.dataSize = item.dataSize;
                outputTensors.emplace_back(tensor);
            }
            Leaks::OpExcuteWatch::GetInstance().OpExcuteEnd(dir, outputTensors, OpType::ATB);
        }
        if (eventType.checkBit(static_cast<size_t>(EventType::LAUNCH_EVENT))) {
            atb::LeaksReportOp(name, params, false);
        }
        if (eventType.checkBit(static_cast<size_t>(EventType::ACCESS_EVENT))) {
            atb::LeaksReportTensors(runnerVariantPack, name);
        }
        return st;
    }

    static bool ReportAtbKernel(std::string &name, const std::string &dirPath)
    {
        auto beforePos = dirPath.find("/before");
        auto afterPos = dirPath.find("/after");
        bool isBeforeLaunch = true;
        std::string path;
        if (beforePos != std::string::npos) {
            path = dirPath.substr(0, beforePos);
        } else if (afterPos != std::string::npos) {
            isBeforeLaunch = false;
            path = dirPath.substr(0, afterPos);
        } else {
            CLIENT_ERROR_LOG("Cannot get kernel path.\n");
            return false;
        }

        AtbKernelRecord record;
        name = path;
        size_t lastSlashPos = name.find_last_of('/');
        if (lastSlashPos != std::string::npos) {
            name = name.substr(lastSlashPos + 1);
        }
        std::ostringstream oss;
        oss << "path:" << dirPath;
        std::string params = oss.str();
        record.eventType = isBeforeLaunch ? KernelEventType::KERNEL_START : KernelEventType::KERNEL_END;
        if (strncpy_s(record.name, sizeof(record.name), name.c_str(), sizeof(record.name) - 1) != EOK) {
            CLIENT_ERROR_LOG("strncpy_s FAILED");
            record.name[0] = '\0';
        }
        if (strncpy_s(record.params, sizeof(record.params), params.c_str(), sizeof(record.params) - 1) != EOK) {
            CLIENT_ERROR_LOG("strncpy_s FAILED");
            record.params[0] = '\0';
        }
        if (!EventReport::Instance(CommType::SOCKET).ReportAtbKernel(record)) {
            CLIENT_ERROR_LOG("Report atb run kernel event failed.\n");
        }
        return true;
    }
    
    void LeaksSaveLaunchParam(const Mki::LaunchParam &launchParam, const std::string &dirPath)
    {
        Config config = EventReport::Instance(CommType::SOCKET).GetConfig();
        BitField<decltype(config.levelType)> levelType(config.levelType);
        if (!levelType.checkBit(static_cast<size_t>(LevelType::LEVEL_KERNEL))) {
            return;
        }
        
        static auto getInTensors = VallinaSymbol<ATBLibLoader>::Instance().Get<Mki::LeaksOriginalGetInTensors>(
            "_ZN3Mki11LaunchParam12GetInTensorsEv");
        static auto getOutTensors = VallinaSymbol<ATBLibLoader>::Instance().Get<Mki::LeaksOriginalGetInTensors>(
            "_ZN3Mki11LaunchParam13GetOutTensorsEv");
        if (getInTensors == nullptr || getOutTensors == nullptr) {
            CLIENT_ERROR_LOG("Cannot find origin function of atb.\n");
            return;
        }

        if (config.watchConfig.isWatched) {
            Leaks::OpExcuteWatch::GetInstance().KernelExcute(dirPath,
                getOutTensors(const_cast<Mki::LaunchParam*>(&launchParam)), OpType::ATB);
        }
        std::string name;
        BitField<decltype(config.eventType)> eventType(config.eventType);
        if (eventType.checkBit(static_cast<size_t>(EventType::LAUNCH_EVENT))) {
            if (!ReportAtbKernel(name, dirPath)) {
                return;
            }
        }

        if (eventType.checkBit(static_cast<size_t>(EventType::ACCESS_EVENT))) {
            atb::LeaksReportTensors(getInTensors, getOutTensors, launchParam, name);
        }
    }
}