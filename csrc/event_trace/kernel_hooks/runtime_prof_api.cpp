// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
#include "runtime_prof_api.h"
#include "client_process.h"
#include "kernel_event_trace.h"
#include "securec.h"

#include <iostream>
namespace Leaks {
int32_t CompactInfoReporterCallbackImpl(uint32_t agingFlag, const void *data, uint32_t length)
{
    if (data == nullptr || length != sizeof(struct MsprofCompactInfo)) {
        CLIENT_ERROR_LOG("Report Compact Info failed with nullptr.");
        return PROFAPI_ERROR;
    }
    const MsprofCompactInfo* compact = reinterpret_cast<const MsprofCompactInfo*>(data);
    if (compact && compact->level == MSPROF_REPORT_RUNTIME_LEVEL && compact->type == RT_PROFILE_TYPE_TASK_TRACK) {
        const MsprofRuntimeTrack &runtimeTrack = compact->data.runtimeTrack;
        auto taskKey = std::make_tuple(runtimeTrack.deviceId, runtimeTrack.streamId,
            static_cast<uint16_t>(runtimeTrack.taskInfo & 0xffff));
        AclnnKernelLaunchMap::GetInstance().AclnnLaunch(taskKey);
    }
    return PROFAPI_ERROR_NONE;
}

void RegisterRtProfileCallback()
{
    using RegisterFunc = int32_t(*)(int32_t, void *, uint32_t);
    auto vallina = VallinaSymbol<RtProfApiLoader>::Instance().Get<RegisterFunc>("MsprofRegisterProfileCallback");
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("get func failed");
        return;
    }

    auto ret = vallina(PROFILE_REPORT_COMPACT_CALLBACK,
        reinterpret_cast<void *>(CompactInfoReporterCallbackImpl), sizeof(void *));
    if (ret != 0) {
        CLIENT_ERROR_LOG("Register rtProfile callback failed.");
    }
    return;
}

void SetProfCommand(uint32_t devId)
{
    CommandHandle command;
    if (memset_s(&command, sizeof(command), 0, sizeof(command)) != EOK) {
        return;
    }

    command.profSwitch = PROF_ACL_API | PROF_TASK_TIME | PROF_RUNTIME_TRACE;
    command.profSwitchHi = 0;
    command.devNums = 1;
    command.devIdList[0] = devId;
    command.modelId = PROF_INVALID_MODE_ID;
    command.type = PROF_COMMANDHANDLE_TYPE_START;

    using ProfSetProfCommandFunc = int32_t(*)(void *, uint32_t);
    auto vallina = VallinaSymbol<RtProfApiLoader>::Instance().Get<ProfSetProfCommandFunc>("profSetProfCommand");
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("Set prof command failed with nullptr.");
        return;
    }

    int ret = vallina(static_cast<void *>(&command), sizeof(CommandHandle));
    if (ret != 0) {
        CLIENT_ERROR_LOG("Set prof command failed.");
        return;
    }

    return;
}
}