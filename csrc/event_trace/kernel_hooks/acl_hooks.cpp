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
#include "serializer.h"
#include "log.h"
#include "record_info.h"
#include "bit_field.h"

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

    // 命令行判断是否为OP级别
    Config userConfig =  EventReport::Instance(CommType::SOCKET).GetConfig();
    BitField<decltype(userConfig.levelType)> levelType(userConfig.levelType);
    if (levelType.checkBit(static_cast<size_t>(LevelType::LEVEL_OP))) {
        if (!Utility::IsPyInterpRepeInited()) {
            CLIENT_ERROR_LOG("Python Interpreter initialization FAILED");
            return ret;
        }
        Utility::PyInterpGuard stat;
        Utility::PythonObject atenCollection = Utility::PythonObject::Import("msleaks.aten_collection", false);
        if (atenCollection.IsBad()) {
            CLIENT_ERROR_LOG("import msleaks.aten_collection FAILED");
            return ret;
        }

        Utility::PythonObject enableAtenCollector = atenCollection.Get("enable_aten_collector");
        if (enableAtenCollector.IsBad()) {
            CLIENT_ERROR_LOG("enable aten collector FAILED");
            return ret;
        }
        enableAtenCollector.Call();
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
