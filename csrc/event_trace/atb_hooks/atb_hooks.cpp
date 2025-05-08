// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "atb_hooks.h"

#include <cstdio>
#include <dlfcn.h>
#include <sstream>
#include <mutex>

#include "event_report.h"
#include "record_info.h"
#include "config_info.h"
#include "bit_field.h"
#include "securec.h"

using namespace Leaks;

namespace atb {
    std::string LeaksGetOpParams(atb::Runner* thisPtr, atb::RunnerVariantPack& runnerVariantPack)
    {
        if (&atb::Runner::GetSaveTensorDir == nullptr) {
            return std::string();
        }

        std::ostringstream oss;
        oss << "{path:" << thisPtr->GetSaveTensorDir() << ","
            << "workspace ptr:" << static_cast<void*>(runnerVariantPack.workspaceBuffer) << ","
            << "workspace size:" << runnerVariantPack.workspaceBufferSize + runnerVariantPack.intermediateBufferSize
            << "}";

        return oss.str();
    }

    std::string LeaksGetKernelParams(const std::string &dirPath)
    {
        std::ostringstream oss;
        oss << "{path:" << dirPath << "}";
        return oss.str();
    }

    atb::Status LeaksRunnerExecute(atb::Runner* thisPtr, atb::RunnerVariantPack& runnerVariantPack)
    {
        if (&atb::Runner::GetOperationName == nullptr) {
            return 0;
        }
        static LeaksOriginalRunnerExecuteFunc originalRunnerExecute = nullptr;
        static std::once_flag initFlag;
        std::call_once(initFlag, []() {
            union {
                void* raw;
                LeaksOriginalRunnerExecuteFunc func;
            } ptr;
            ptr.raw = dlsym(RTLD_NEXT, "_ZN3atb6Runner7ExecuteERNS_17RunnerVariantPackE");
            originalRunnerExecute = ptr.func;       // dlsym方法返回指针是void*类型，使用union进行转换
        });
        if (!originalRunnerExecute) {
            CLIENT_ERROR_LOG("Cannot find origin function of atb.\n");
            return 0;
        }
        Config config = EventReport::Instance(CommType::SOCKET).GetConfig();
        BitField<decltype(config.levelType)> levelType(config.levelType);
        if (!levelType.checkBit(static_cast<size_t>(LevelType::LEVEL_OP))) {
            return originalRunnerExecute(thisPtr, runnerVariantPack);
        }

        AtbOpExecuteRecord record;
        record.eventType = OpEventType::ATB_START;
        if (strncpy_s(record.name, sizeof(record.name),
            thisPtr->GetOperationName().c_str(), sizeof(record.name) - 1) != EOK) {
            strncpy_s(record.name, sizeof(record.name), "unknown op", sizeof(record.name) - 1);
            CLIENT_ERROR_LOG("strncpy_s FAILED");
        }
        std::string params = atb::LeaksGetOpParams(thisPtr, runnerVariantPack);
        if (strncpy_s(record.params, sizeof(record.params), params.c_str(), sizeof(record.params) - 1) != EOK) {
            CLIENT_ERROR_LOG("strncpy_s FAILED");
        }
        if (!EventReport::Instance(CommType::SOCKET).ReportAtbOpExecute(record)) {
            CLIENT_ERROR_LOG("Report atb op start event failed.\n");
        }

        atb::Status st = originalRunnerExecute(thisPtr, runnerVariantPack);
        record.eventType = OpEventType::ATB_END;
        if (!EventReport::Instance(CommType::SOCKET).ReportAtbOpExecute(record)) {
            CLIENT_ERROR_LOG("Report atb op end event failed.\n");
        }
        return st;
    }

    void LeaksSaveLaunchParam(const Mki::LaunchParam &launchParam, const std::string &dirPath)
    {
        if (&Mki::LaunchParam::GetInTensors == nullptr || &Mki::LaunchParam::GetOutTensors == nullptr) {
            return;
        }

        Config config = EventReport::Instance(CommType::SOCKET).GetConfig();
        BitField<decltype(config.levelType)> levelType(config.levelType);
        if (!levelType.checkBit(static_cast<size_t>(LevelType::LEVEL_KERNEL))) {
            return;
        }

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
            return;
        }

        std::string name = path;
        size_t lastSlashPos = name.find_last_of('/');
        if (lastSlashPos != std::string::npos) {
            name = name.substr(lastSlashPos + 1);
        }

        AtbKernelRecord record;
        record.eventType = isBeforeLaunch ? KernelEventType::KERNEL_START : KernelEventType::KERNEL_END;
        if (strncpy_s(record.name, sizeof(record.name), name.c_str(), sizeof(record.name) - 1) != EOK) {
            strncpy_s(record.name, sizeof(record.name), "unknown kernel", sizeof(record.name) - 1);
            CLIENT_ERROR_LOG("strncpy_s FAILED");
        }
        std::string params = atb::LeaksGetKernelParams(path);
        if (strncpy_s(record.params, sizeof(record.params), params.c_str(), sizeof(record.params) - 1) != EOK) {
            CLIENT_ERROR_LOG("strncpy_s FAILED");
        }
        if (!EventReport::Instance(CommType::SOCKET).ReportAtbKernel(record)) {
            CLIENT_ERROR_LOG("Report atb run kernel event failed.\n");
        }
    }
}