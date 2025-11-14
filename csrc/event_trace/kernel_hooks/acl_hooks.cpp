// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

#include "acl_hooks.h"
#include <cstdint>
#include <vector>
#include <algorithm>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <unordered_map>

#include "cpython.h"
#include "event_report.h"
#include "vallina_symbol.h"
#include "log.h"
#include "record_info.h"
#include "bit_field.h"
#include "trace_manager/event_trace_manager.h"

using namespace Leaks;
 
ACL_FUNC_VISIBILITY aclError aclInit(const char *configPath)
{
    using AclInit = decltype(&aclInit);
    auto vallina = VallinaSymbol<AclLibLoader>::Instance().Get<AclInit>("aclInit");
    if (vallina == nullptr) {
        LOG_ERROR("vallina func get FAILED: " + std::string(__func__));
        return ACL_ERROR_INTERNAL_ERROR;
    }

    aclError ret = vallina(configPath);

    EventTraceManager::Instance().SetAclInitStatus(true);

    if (!EventTraceManager::Instance().IsTracingEnabled()) {
        return ret;
    }

    if (!EventReport::Instance(LeaksCommType::SHARED_MEMORY).ReportAclItf(RecordSubType::INIT)) {
        LOG_ERROR("aclInit report FAILED");
    }
    if (BitPresent(GetConfig().analysisType, static_cast<size_t>(AnalysisType::DECOMPOSE_ANALYSIS))) {
        Utility::LeaksPythonCall("msleaks.optimizer_step_hook", "enable_optimizer_step_hook");
    }

    return ret;
}

ACL_FUNC_VISIBILITY aclError aclFinalize()
{
    using AclFinalize = decltype(&aclFinalize);
    auto vallina = VallinaSymbol<AclLibLoader>::Instance().Get<AclFinalize>("aclFinalize");
    if (vallina == nullptr) {
        LOG_ERROR("vallina func get FAILED: " + std::string(__func__));
        return ACL_ERROR_INTERNAL_ERROR;
    }

    aclError ret = vallina();

    if (!EventTraceManager::Instance().IsTracingEnabled()) {
        return ret;
    }

    if (!EventReport::Instance(LeaksCommType::SHARED_MEMORY).ReportAclItf(RecordSubType::FINALIZE)) {
        LOG_ERROR("aclInit report FAILED");
    }

    if (BitPresent(GetConfig().analysisType, static_cast<size_t>(AnalysisType::DECOMPOSE_ANALYSIS))) {
        Utility::LeaksPythonCall("msleaks.optimizer_step_hook", "disable_optimizer_step_hook");
    }

    return ret;
}
