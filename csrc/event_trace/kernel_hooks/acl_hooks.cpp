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

#include "event_report.h"
#include "vallina_symbol.h"
#include "serializer.h"
#include "log.h"
#include "record_info.h"

using namespace Leaks;

ACL_FUNC_VISIBILITY aclError aclInit(const char *configPath)
{
    using AclInit = decltype(&aclInit);
    auto vallina = VallinaSymbol<AclLibLoader>::Instance().Get<AclInit>("aclInit");
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__));
        return ACL_ERROR_INTERNAL_ERROR;
    }

    aclError ret = vallina(configPath);
    if (!EventReport::Instance(CommType::SOCKET).ReportAclItf(AclOpType::INIT)) {
        CLIENT_ERROR_LOG("aclInit report FAILED");
    }
    return ret;
}

ACL_FUNC_VISIBILITY aclError aclFinalize()
{
    using AclFinalize = decltype(&aclFinalize);
    auto vallina = VallinaSymbol<AclLibLoader>::Instance().Get<AclFinalize>("aclFinalize");
    if (vallina == nullptr) {
        CLIENT_ERROR_LOG("vallina func get FAILED: " + std::string(__func__));
        return ACL_ERROR_INTERNAL_ERROR;
    }

    aclError ret = vallina();
    if (!EventReport::Instance(CommType::SOCKET).ReportAclItf(AclOpType::FINALIZE)) {
        CLIENT_ERROR_LOG("aclInit report FAILED");
    }
    return ret;
}
